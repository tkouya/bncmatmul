/********************************************************************************/
/*                                                                              */
/* bncomp_sparse.c : Parallelized Sparse Matrix and Vector Library              */
/* Copyright (c) 2024 Tomonori Kouya, All rights reserved.                      */
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
#include <stdlib.h>
#include <math.h>

// Sparse linear computation
#include "bncsparse.h"

// Parallel computing using OpenMP
#include "bncomp.h"

//-----------------------------------------------
// Double precision
//-----------------------------------------------

/* Multiply DRSMatrix * DVector */
int _bncomp_mul_drsmatrix_dvec(DVector ret, DRSMatrix mat, DVector vec)
{
	long int i, j, total_index;
	double tmp, mat_ij, vec_j, ret_i;
	int thread_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	set0_dvector(ret);

#ifdef USE_IMKL
	#ifdef USE_IMKL_OLD // 2025-07-10(Thu)
	//#if 0
    int *i_mat_csr, *j_mat_csr, row_dim = (int)mat->row_dim;

	// convert our CSR to intel math kernel csr format
	i_mat_csr = (int *)calloc(mat->row_dim + 1, sizeof(int));
	j_mat_csr = (int *)calloc(mat->real_nzero_total_num, sizeof(int));

	total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		i_mat_csr[i] = (int)total_index;
		//if(a->nzero_col_dim[i] >= 1)
		if(mat->real_nzero_col_dim[i] >= 1)
		{
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{
				j_mat_csr[total_index] = mat->nzero_index[i][j];
				total_index++;
			}
			// embed gap among nzero_col_dim and real_nzero_col_dim
			for(j = mat->nzero_col_dim[i]; j < mat->real_nzero_col_dim[i]; j++)
			{
				j_mat_csr[total_index] = mat->nzero_index[i][mat->nzero_col_dim[i] - 1] + (j + 1 - mat->nzero_col_dim[i]);
				total_index++;
			}
		}
	}
	i_mat_csr[row_dim] = mat->real_nzero_total_num;
	mkl_cspblas_dcsrgemv("N", &row_dim, mat->element, i_mat_csr, j_mat_csr, vec->element, ret->element);

	free(i_mat_csr); free(j_mat_csr);
//#endif // 0
	#else // USE_IMKL new
		sparse_matrix_t mkl_drsmat;
		MKL_INT *i_csr_start, *i_csr_end, *j_csr;
		struct matrix_descr descr;

		// DRSMatrix -> MKL_CSRmatrix
		convert_drsmatrix_mkl_csrmat(&mkl_drsmat, i_csr_start, i_csr_end, j_csr, mat);

		// b := A * x by IMKL
		descr.type = SPARSE_MATRIX_TYPE_GENERAL;
		//descr.mode = ?
		//descr.diag = ?
		mkl_sparse_d_mv(
			SPARSE_OPERATION_NON_TRANSPOSE,
			(double)1.0,
			mkl_drsmat,
			descr,
			vec->element,
			(double)0.0,
			vec->element
		);
	    free(i_csr_start); free(i_csr_end); free(j_csr);
		mkl_sparse_destroy(mkl_drsmat);
	#endif // USE_IMKL_OLD
#else // USE_IMKL

	//total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, vb4, tmp4;
	long int jmax, jres, col_dim;
	double mat_vec_i;

	//#pragma omp parallel for schedule(static) private(thread_index, j, total_index,a4, vb4, tmp4, tmp4_mul, jmax, jres, col_dim, mat_vec_i, vset) threadprivate(thread_index, j, total_index,a4, vb4, tmp4, tmp4_mul, jmax, jres, col_dim, mat_vec_i, vset)
	#pragma omp parallel for schedule(static) private(thread_index, j, total_index,a4, vb4, tmp4, jmax, jres, col_dim, mat_vec_i)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//ret->element[i] = 0.0;
		tmp4 = _mm256_setzero_pd();
		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a4  = _mm256_load_pd(&(mat->element[total_index]));
			vb4 = _mm256_set_pd(
				vec->element[mat->nzero_index[i][j + 3]],
				vec->element[mat->nzero_index[i][j + 2]],
				vec->element[mat->nzero_index[i][j + 1]],
				vec->element[mat->nzero_index[i][j    ]]
			);
			tmp4 = _mm256_fmadd_pd(a4, vb4, tmp4);

			// total_index++;
			total_index += _BNC_D_WIDTH;
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];
		mat_vec_i = _bncavx2_dsum256d(tmp4); // tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];
		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);
		col_dim = mat->nzero_col_dim[i];
		switch(jres)
		{
			case 1: 
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 2:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 3:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
		}
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		set_dvector_i(ret, i, mat_vec_i);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, vb8, tmp8;
	long int jmax, jres, col_dim;
	double mat_vec_i;

	//#pragma omp parallel for schedule(static) private(thread_index, j, total_index,a8, vb8, tmp8, tmp8_mul, jmax, jres, col_dim, mat_vec_i, vset) threadprivate(thread_index, j, total_index,a8, vb8, tmp8, tmp8_mul, jmax, jres, col_dim, mat_vec_i, vset)
	#pragma omp parallel for schedule(static) private(thread_index, j, total_index,a8, vb8, tmp8, jmax, jres, col_dim, mat_vec_i)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//ret->element[i] = 0.0;
		tmp8 = _mm512_setzero_pd();
		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres = mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a8  = _mm512_load_pd(&(mat->element[total_index]));
			vb8 = _mm512_set_pd(
				vec->element[mat->nzero_index[i][j + 7]],
				vec->element[mat->nzero_index[i][j + 6]],
				vec->element[mat->nzero_index[i][j + 5]],
				vec->element[mat->nzero_index[i][j + 4]],
				vec->element[mat->nzero_index[i][j + 3]],
				vec->element[mat->nzero_index[i][j + 2]],
				vec->element[mat->nzero_index[i][j + 1]],
				vec->element[mat->nzero_index[i][j    ]]
			);
			tmp8 = _mm512_fmadd_pd(a8, vb8, tmp8);

			// total_index++;
			total_index += _BNC_D_WIDTH;
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];
		mat_vec_i = _bncavx2_dsum512d(tmp8);
		col_dim = mat->nzero_col_dim[i];
		switch(jres)
		{
			case 1: 
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_col_dim[i] - 1];
				break; 
			case 2:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 3:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 4:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 4]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 5:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 5]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 4]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 6:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 6]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 5]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 4]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 7:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 7]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 6]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 5]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 4]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
		}
		set_dvector_i(ret, i, mat_vec_i);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP plain-double SpMV: each row is processed by one thread,
	 * inner loop uses VL-agnostic hardware gather and predicated load/store. */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel for schedule(static) private(thread_index, j, total_index)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k;
			thread_index = omp_get_thread_num();
			total_index = 0;
			for(_k = 0; _k < i; _k++) total_index += mat->real_nzero_col_dim[_k];

			svfloat64_t tmp_v = svdup_n_f64(0.0);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / _vl) * _vl;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += _vl)
			{
				svbool_t pg = svptrue_b64();
				svfloat64_t a_v  = svld1_f64(pg, &(mat->element[total_index]));
				svint64_t   idx  = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
				svfloat64_t vb_v = svld1_gather_s64index_f64(pg, vec->element, idx);
				tmp_v = svmla_f64_x(pg, tmp_v, a_v, vb_v);
				total_index += _vl;
			}
			double mat_vec_i = svaddv_f64(svptrue_b64(), tmp_v);
			long int col_dim = mat->nzero_col_dim[i];
			long int _t;
			for(_t = 0; _t < jres; _t++)
			{
				mat_vec_i += mat->element[total_index] *
				             vec->element[mat->nzero_index[i][col_dim - jres + _t]];
				total_index++;
			}
			set_dvector_i(ret, i, mat_vec_i);
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel for schedule(static) private(thread_index, j, total_index)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k;
			thread_index = omp_get_thread_num();
			total_index = 0;
			for(_k = 0; _k < i; _k++) total_index += mat->real_nzero_col_dim[_k];

			float64x2_t tmp2 = vdupq_n_f64(0.0);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / 2) * 2;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += 2)
			{
				float64x2_t a2 = vld1q_f64(&(mat->element[total_index]));
				float64x2_t vb2 = (float64x2_t){
					vec->element[mat->nzero_index[i][j    ]],
					vec->element[mat->nzero_index[i][j + 1]]
				};
				tmp2 = vfmaq_f64(tmp2, a2, vb2);
				total_index += 2;
			}
			double mat_vec_i = vgetq_lane_f64(tmp2, 0) + vgetq_lane_f64(tmp2, 1);
			long int col_dim = mat->nzero_col_dim[i];
			long int _t;
			for(_t = 0; _t < jres; _t++)
			{
				mat_vec_i += mat->element[total_index] *
				             vec->element[mat->nzero_index[i][col_dim - jres + _t]];
				total_index++;
			}
			set_dvector_i(ret, i, mat_vec_i);
		}
	}

#else // others
	#pragma omp parallel for schedule(static) private(thread_index, j, total_index)
	for(i = 0; i < mat->row_dim; i++)
	{
		long int _k;
		/* compute per-row offset; private(total_index) is uninitialized otherwise */
		total_index = 0;
		for(_k = 0; _k < i; _k++) total_index += mat->real_nzero_col_dim[_k];
		ret->element[i] = 0.0;
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			total_index++;
		}
	}
#endif // __AVX2__
#endif // USE_IMKL

	return SUCCESS;
}

/* Multiply DRSMatrix^T * DVector */
int _bncomp_mul_drsmatrixt_dvec(DVector ret, DRSMatrix mat, DVector vec)
{
	long int i, j, total_index;
	double tmp, mat_ji, vec_j, ret_i;
	int thread_index, num_threads;
	long int div_row, start_i, end_i;
	DVector work_vec[BNCOMP_MAX_NUM_THREADS];

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

//#if 0
#ifdef USE_IMKL
	#ifdef USE_IMKL_OLD
    int *i_mat_csr, *j_mat_csr, row_dim = (int)mat->row_dim;

	// convert our CSR to intel math kernel csr format
	i_mat_csr = (int *)calloc(mat->row_dim + 1, sizeof(int));
	j_mat_csr = (int *)calloc(mat->real_nzero_total_num, sizeof(int));

	total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		i_mat_csr[i] = (int)total_index;
		//if(a->nzero_col_dim[i] >= 1)
		if(mat->real_nzero_col_dim[i] >= 1)
		{
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{
				j_mat_csr[total_index] = mat->nzero_index[i][j];
				total_index++;
			}
			// embed gap among nzero_col_dim and real_nzero_col_dim
			for(j = mat->nzero_col_dim[i]; j < mat->real_nzero_col_dim[i]; j++)
			{
				j_mat_csr[total_index] = mat->nzero_index[i][mat->nzero_col_dim[i] - 1] + (j + 1 - mat->nzero_col_dim[i]);
				total_index++;
			}
		}
	}
	i_mat_csr[row_dim] = mat->real_nzero_total_num;
	mkl_cspblas_dcsrgemv("T", &row_dim, mat->element, i_mat_csr, j_mat_csr, vec->element, ret->element);

	free(i_mat_csr); free(j_mat_csr);
	#else // USE_IMKL_OLD
		sparse_matrix_t mkl_drsmat;
		MKL_INT *i_csr_start, *i_csr_end, *j_csr;
		struct matrix_descr descr;

		// DRSMatrix -> MKL_CSRmatrix
		convert_drsmatrix_mkl_csrmat(&mkl_drsmat, i_csr_start, i_csr_end, j_csr, mat);

		// b := A * x by IMKL
		descr.type = SPARSE_MATRIX_TYPE_GENERAL;
		//descr.mode = ?
		//descr.diag = ?
		mkl_sparse_d_mv(
			SPARSE_OPERATION_TRANSPOSE,
			(double)1.0,
			mkl_drsmat,
			descr,
			vec->element,
			(double)0.0,
			vec->element
		);
	    free(i_csr_start); free(i_csr_end); free(j_csr);
	#endif // USE_IMKL_OLD
#else // USE_IMKL
//#endif // 0

	total_index = 0;
	set0_dvector(ret);

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, vb4, ret4;
	long int jmax, jres, col_dim;

	//#pragma omp parallel for schedule(static) threadprivate(thread_index, j, total_index,a4, vb4, tmp4, tmp4_mul, jmax, jres, col_dim, mat_vec_i, vset)
	#pragma omp parallel for schedule(static) private(thread_index, j, total_index,a4, vb4, jmax, jres, col_dim)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		vb4 = _mm256_set1_pd(vec->element[i]);
		jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		//jmax = (mat->row_dim / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->row_dim % _BNC_D_WIDTH;
		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			a4  = _mm256_load_pd(&(mat->element[total_index]));
			ret4 = _mm256_set_pd(
				ret->element[mat->nzero_index[i][j + 3]],
				ret->element[mat->nzero_index[i][j + 2]],
				ret->element[mat->nzero_index[i][j + 1]],
				ret->element[mat->nzero_index[i][j    ]]
			);
			ret4 = _mm256_fmadd_pd(a4, vb4, ret4);

			ret->element[mat->nzero_index[i][j + 3]] = ret4[3];
			ret->element[mat->nzero_index[i][j + 2]] = ret4[2];
			ret->element[mat->nzero_index[i][j + 1]] = ret4[1];
			ret->element[mat->nzero_index[i][j    ]] = ret4[0];

			// total_index++;
			//total_index += _BNC_D_WIDTH;
		}
		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);
		col_dim = mat->nzero_col_dim[i];
		//col_dim = mat->real_nzero_col_dim[i];
		switch(jres)
		{
			case 1: 
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 2:
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 3:
				ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
		}
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, vb8, ret8;
	long int jmax, jres, col_dim;

	//#pragma omp parallel for schedule(static) threadprivate(thread_index, j, total_index,a8, vb8, tmp8, jmax, jres, col_dim, mat_vec_i, vset)
	#pragma omp parallel for schedule(static) private(thread_index, j, total_index,a8, vb8, jmax, jres, col_dim)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		vb8 = _mm512_set1_pd(vec->element[i]);
		jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;

		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			a8  = _mm512_load_pd(&(mat->element[total_index]));
			ret8 = _mm512_set_pd(
				ret->element[mat->nzero_index[i][j + 7]],
				ret->element[mat->nzero_index[i][j + 6]],
				ret->element[mat->nzero_index[i][j + 5]],
				ret->element[mat->nzero_index[i][j + 4]],
				ret->element[mat->nzero_index[i][j + 3]],
				ret->element[mat->nzero_index[i][j + 2]],
				ret->element[mat->nzero_index[i][j + 1]],
				ret->element[mat->nzero_index[i][j    ]]
			);
			ret8 = _mm512_fmadd_pd(a8, vb8, ret8);

			ret->element[mat->nzero_index[i][j + 7]] = ret8[7];
			ret->element[mat->nzero_index[i][j + 6]] = ret8[6];
			ret->element[mat->nzero_index[i][j + 5]] = ret8[5];
			ret->element[mat->nzero_index[i][j + 4]] = ret8[4];
			ret->element[mat->nzero_index[i][j + 3]] = ret8[3];
			ret->element[mat->nzero_index[i][j + 2]] = ret8[2];
			ret->element[mat->nzero_index[i][j + 1]] = ret8[1];
			ret->element[mat->nzero_index[i][j    ]] = ret8[0];

			// total_index++;
			total_index += _BNC_D_WIDTH;
		}
		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);
		//col_dim = mat->nzero_col_dim[i];
		col_dim = mat->real_nzero_col_dim[i];
		switch(jres)
		{
			case 1: 
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 2:
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 3:
				ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 4:
				ret->element[mat->nzero_index[i][col_dim - 4]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 5:
				ret->element[mat->nzero_index[i][col_dim - 5]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 4]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 6:
				ret->element[mat->nzero_index[i][col_dim - 6]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 5]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 4]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 7:
				ret->element[mat->nzero_index[i][col_dim - 7]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 6]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 5]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 4]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break;

		}
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP A^T*x (plain double): each thread accumulates into its own
	 * work_vec (race-free), reduced into ret afterwards; inner loop uses
	 * hardware gather/scatter on the private work_vec. */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			long int _k, _ti;
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			work_vec[thread_index] = init_dvector(vec->dim);
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;
			for(i = start_i; i < end_i; i++)
			{
				double *w = work_vec[thread_index]->element;
				svfloat64_t vb_v = svdup_n_f64(vec->element[i]);
				_ti = 0;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / _vl) * _vl;
				long int jres = nz - jmax;
				for(j = 0; j < jmax; j += _vl)
				{
					svbool_t pg = svptrue_b64();
					svfloat64_t a_v  = svld1_f64(pg, &(mat->element[_ti]));
					svint64_t   idx  = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
					svfloat64_t cur  = svld1_gather_s64index_f64(pg, w, idx);
					svfloat64_t newv = svmla_f64_x(pg, cur, a_v, vb_v);
					svst1_scatter_s64index_f64(pg, w, idx, newv);
					_ti += _vl;
				}
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					w[mat->nzero_index[i][col_dim - jres + _t]]
					    += mat->element[_ti] * vec->element[i];
					_ti++;
				}
			}
		}
		#pragma omp parallel private(thread_index)
		{
			thread_index = omp_get_thread_num();
			#pragma omp critical
			{ add_dvector(ret, ret, work_vec[thread_index]); free_dvector(work_vec[thread_index]); }
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			long int _k, _ti;
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			work_vec[thread_index] = init_dvector(vec->dim);
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;
			for(i = start_i; i < end_i; i++)
			{
				double *w = work_vec[thread_index]->element;
				float64x2_t vb2 = vdupq_n_f64(vec->element[i]);
				_ti = 0;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / 2) * 2;
				long int jres = nz - jmax;
				for(j = 0; j < jmax; j += 2)
				{
					float64x2_t a2 = vld1q_f64(&(mat->element[_ti]));
					float64x2_t ret2 = (float64x2_t){
						w[mat->nzero_index[i][j    ]],
						w[mat->nzero_index[i][j + 1]]
					};
					ret2 = vfmaq_f64(ret2, a2, vb2);
					w[mat->nzero_index[i][j    ]] = vgetq_lane_f64(ret2, 0);
					w[mat->nzero_index[i][j + 1]] = vgetq_lane_f64(ret2, 1);
					_ti += 2;
				}
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					w[mat->nzero_index[i][col_dim - jres + _t]]
					    += mat->element[_ti] * vec->element[i];
					_ti++;
				}
			}
		}
		#pragma omp parallel private(thread_index)
		{
			thread_index = omp_get_thread_num();
			#pragma omp critical
			{ add_dvector(ret, ret, work_vec[thread_index]); free_dvector(work_vec[thread_index]); }
		}
	}

#else // __AVX2__
	/* scalar + OpenMP A^T*x: per-thread work_vec accumulation, race-free.
	 * (The previous in-place scatter left total_index uninitialised under
	 *  private() and raced on ret; this version fixes both.) */
	{
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			long int _k, _ti;
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			work_vec[thread_index] = init_dvector(vec->dim);
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;
			for(i = start_i; i < end_i; i++)
			{
				double *w = work_vec[thread_index]->element;
				_ti = 0;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				for(j = 0; j < mat->nzero_col_dim[i]; j++)
				{
					w[mat->nzero_index[i][j]] += mat->element[_ti] * vec->element[i];
					_ti++;
				}
			}
		}
		#pragma omp parallel private(thread_index)
		{
			thread_index = omp_get_thread_num();
			#pragma omp critical
			{ add_dvector(ret, ret, work_vec[thread_index]); free_dvector(work_vec[thread_index]); }
		}
	}
#endif // __AVX2__
#endif // USE_IMKL

	return SUCCESS;
}

/* Multiply CDRSMatrix * CDVector */
int _bncomp_mul_cdrsmatrix_cdvec(CDVector ret, CDRSMatrix mat, CDVector vec)
{
	long int i, j, total_index;
	DVector in_vec_re, in_vec_im, in_ret_re, in_ret_im;
	DVector tmp_vec[4];

	if((ret->dim < mat->re->col_dim) || (vec->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	// vec_re * vec_im * i = vec
	in_vec_re = init_dvector(vec->dim);
	in_vec_im = init_dvector(vec->dim);
	separate_cdvector(in_vec_re, in_vec_im, vec);

	in_ret_re = init_dvector(mat->re->row_dim);
	in_ret_im = init_dvector(mat->re->row_dim);
	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_dvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_drsmatrix_dvec(tmp_vec[0], mat->re, in_vec_re);
	_bncomp_mul_drsmatrix_dvec(tmp_vec[1], mat->re, in_vec_im);
	_bncomp_mul_drsmatrix_dvec(tmp_vec[2], mat->im, in_vec_re);
	_bncomp_mul_drsmatrix_dvec(tmp_vec[3], mat->im, in_vec_im);

	_bncomp_sub_dvector(in_ret_re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_dvector(in_ret_im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	free_dvector(in_vec_re);
	free_dvector(in_vec_im);
	free_dvector(in_ret_re);
	free_dvector(in_ret_im);
	for(i = 0; i < 4; i++)
		free_dvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CDRSMatrix^T * CDVector */
int _bncomp_mul_cdrsmatrixt_cdvec(CDVector ret, CDRSMatrix mat, CDVector vec)
{
	long int i, j, total_index;
	DVector in_vec_re, in_vec_im, in_ret_re, in_ret_im;
	DVector tmp_vec[4];

	if((ret->dim < mat->re->col_dim) || (vec->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	// vec_re * vec_im * i = vec
	in_vec_re = init_dvector(mat->re->row_dim);
	in_vec_im = init_dvector(mat->re->row_dim);
	separate_cdvector(in_vec_re, in_vec_im, vec);

	in_ret_re = init_dvector(vec->dim);
	in_ret_im = init_dvector(vec->dim);
	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_dvector(vec->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_drsmatrixt_dvec(tmp_vec[0], mat->re, in_vec_re);
	_bncomp_mul_drsmatrixt_dvec(tmp_vec[1], mat->re, in_vec_im);
	_bncomp_mul_drsmatrixt_dvec(tmp_vec[2], mat->im, in_vec_re);
	_bncomp_mul_drsmatrixt_dvec(tmp_vec[3], mat->im, in_vec_im);

	_bncomp_sub_dvector(in_ret_re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_dvector(in_ret_im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	free_dvector(in_vec_re);
	free_dvector(in_vec_im);
	free_dvector(in_ret_re);
	free_dvector(in_ret_im);

	// unknown error
	// 2024-09-24 T.Kouya
	for(i = 0; i < 4; i++)
		free_dvector(tmp_vec[i]);
	//for(i = 0; i < 4; i++)
	//	free(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply conj(CDRSMatrix)^T * CDVector */
int _bncomp_mul_cdrsmatrixs_cdvec(CDVector ret, CDRSMatrix mat, CDVector vec)
{
	long int i, j, total_index;
	DVector in_vec_re, in_vec_im, in_ret_re, in_ret_im;
	DVector tmp_vec[4];

	if((ret->dim < mat->re->col_dim) || (vec->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	// vec_re * vec_im * i = vec
	in_vec_re = init_dvector(mat->re->row_dim);
	in_vec_im = init_dvector(mat->re->row_dim);
	separate_cdvector(in_vec_re, in_vec_im, vec);

	in_ret_re = init_dvector(vec->dim);
	in_ret_im = init_dvector(vec->dim);
	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_dvector(vec->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_drsmatrixt_dvec(tmp_vec[0], mat->re, in_vec_re);
	_bncomp_mul_drsmatrixt_dvec(tmp_vec[1], mat->re, in_vec_im);
	_bncomp_mul_drsmatrixt_dvec(tmp_vec[2], mat->im, in_vec_re);
	_bncomp_mul_drsmatrixt_dvec(tmp_vec[3], mat->im, in_vec_im);

	// mat_re * vec_re + mat_im * vec_im
	_bncomp_add_dvector(in_ret_re, tmp_vec[0], tmp_vec[3]);
	// mat_re * vec_im - mat_im * vec_re
	_bncomp_sub_dvector(in_ret_im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	free_dvector(in_vec_re);
	free_dvector(in_vec_im);
	free_dvector(in_ret_re);
	free_dvector(in_ret_im);

	// unknown error
	// 2024-09-24 T.Kouya
	for(i = 0; i < 4; i++)
		free_dvector(tmp_vec[i]);
	//for(i = 0; i < 4; i++)
	//	free(tmp_vec[i]);

	return SUCCESS;
}

//-----------------------------------------------
// DD precision
//-----------------------------------------------

/* Multiply DDRSMatrix * DDVector */
int _bncomp_mul_ddrsmatrix_ddvec(DDVector ret, DDRSMatrix mat, DDVector vec)
{
	long int i, j, total_index; // total_index[BNCOMP_MAX_NUM_THREADS];
	//ddfloat tmp, mat_ij, vec_j, ret_i;
	//double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE], mat_ij[BNCOMP_MAX_NUM_THREADS][DDSIZE], vec_j[BNCOMP_MAX_NUM_THREADS][DDSIZE], ret_i[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	double tmp[DDSIZE], mat_ij[DDSIZE], vec_j[DDSIZE], ret_i[DDSIZE];
	int thread_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	//__m256d a4[BNCOMP_MAX_NUM_THREADS][DDSIZE], vb4[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp4_mul[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	//long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	//double mat_vec_i[BNCOMP_MAX_NUM_THREADS][DDSIZE], vset[BNCOMP_MAX_NUM_THREADS][_BNC_D_WIDTH];
	__m256d a4[DDSIZE], vb4[DDSIZE], tmp4[DDSIZE], tmp4_mul[DDSIZE];
	long int jmax, jres, col_dim;
	double mat_vec_i[DDSIZE], vset[_BNC_D_WIDTH];

    //#pragma omp parallel for private(thread_index, j)
	//#pragma omp parallel for schedule(guided) private(thread_index, j, total_index,a4, vb4, tmp4, tmp4_mul, jmax, jres, col_dim, mat_vec_i, vset)
	//#pragma omp parallel for schedule(dynamic) private(thread_index, j, total_index,a4, vb4, tmp4, tmp4_mul, jmax, jres, col_dim, mat_vec_i, vset)
	//#pragma omp parallel for schedule(static) private(thread_index, j, total_index,a4, vb4, tmp4, tmp4_mul, jmax, jres, col_dim, mat_vec_i, vset)
	#pragma omp parallel for schedule(static) private(thread_index, j, total_index,a4, vb4, tmp4, tmp4_mul, jmax, jres, col_dim, mat_vec_i, vset)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//ret->element[i] = 0.0;
		//tmp4 = _mm256_setzero_pd();
		//_bncavx2_set0_dd(tmp4[thread_index]);
		_bncavx2_set0_dd(tmp4);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jmax = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		//jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;
		jres =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

        //total_index[thread_index] = 0;
        total_index = 0;
        //for(j = 0; j < i; j++) total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;
        for(j = 0; j < i; j++) total_index += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		//printf("%ld %ld jmax, jrec = %ld, %ld\n", i, total_index, jmax, jres);
		//for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//printf("load0 ");fflush(stdout);
			//a4[thread_index][0]  = _mm256_load_pd(&(mat->element[0][total_index[thread_index]]));
			a4[0]  = _mm256_load_pd(&(mat->element[0][total_index]));			//printf("load1 ");fflush(stdout);
			//a4[thread_index][1]  = _mm256_load_pd(&(mat->element[1][total_index[thread_index]]));
			a4[1]  = _mm256_load_pd(&(mat->element[1][total_index]));

			//printf("set0 "); 
			// %ld, %ld, %ld, %ld ", mat->nzero_index[i][j+3], mat->nzero_index[i][j+2], mat->nzero_index[i][j+1], mat->nzero_index[i][j]); fflush(stdout)
			//vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			//if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[0][mat->nzero_index[i][j    ]];
			//if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[0][mat->nzero_index[i][j + 1]];
			//if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[0][mat->nzero_index[i][j + 2]];
			//if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[0][mat->nzero_index[i][j + 3]];
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[0][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[0][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = vec->element[0][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = vec->element[0][mat->nzero_index[i][j + 3]];
			//printf("set0 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			/* vb4[thread_index][0] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[0][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[0][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[0][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[0][mat->nzero_index[i][j    ]]
			); */
			vb4[0] = _mm256_set_pd(
				vset[3], // vec->element[0][mat->nzero_index[i][j + 3]],
				vset[2], // vec->element[0][mat->nzero_index[i][j + 2]],
				vset[1], // vec->element[0][mat->nzero_index[i][j + 1]],
				vset[0]  // vec->element[0][mat->nzero_index[i][j    ]]
			);
			//printf("set1 ");
			//vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			/*
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[1][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[1][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[1][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[1][mat->nzero_index[i][j + 3]];
			*/
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[1][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[1][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = vec->element[1][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = vec->element[1][mat->nzero_index[i][j + 3]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			/* vb4[thread_index][1] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[1][mat->nzero_index[i][j    ]]
			); */
			vb4[1] = _mm256_set_pd(
				vset[3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);			//printf("mul_add ");
			//tmp4 = _mm256_fmadd_pd(a4, vb4, tmp4);
			//_bncavx2_rdd_mul(tmp4_mul[thread_index], a4[thread_index], vb4[thread_index]);
			//_bncavx2_rdd_add(tmp4[thread_index], tmp4[thread_index], tmp4_mul[thread_index]);
			_bncavx2_rdd_mul(tmp4_mul, a4, vb4);
			_bncavx2_rdd_add(tmp4, tmp4, tmp4_mul);
			// total_index++;
			//total_index[thread_index] += _BNC_D_WIDTH;
			total_index += _BNC_D_WIDTH;
			//printf("%ld ", j);

			//printf("%ld ", j);
		}
		//printf("\n");
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];
		//_bncavx2_rdd_sum256d(mat_vec_i[thread_index], tmp4[thread_index]);
		_bncavx2_rdd_sum256d(mat_vec_i, tmp4);

		//printf("%ld %ld mat_vec_i = %25.17e, %ld\n", i, total_index, mat_vec_i[0], mat->nzero_col_dim[i]);
		//col_dim[thread_index] = mat->nzero_col_dim[i];
		//set_ddvector_i(ret, i, mat_vec_i[thread_index]);
		col_dim = mat->nzero_col_dim[i];
		set_ddvector_i(ret, i, mat_vec_i);
	}

#elif defined(__AVX512F__) // __AVX512F__
	//__m512d a8[BNCOMP_MAX_NUM_THREADS][DDSIZE], vb8[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp8_mul[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	//long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	//double mat_vec_i[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	__m512d a8[DDSIZE], vb8[DDSIZE], tmp8[DDSIZE], tmp8_mul[DDSIZE];
	long int jmax, jres, col_dim;
	double mat_vec_i[DDSIZE];

    //#pragma omp parallel for schedule(dynamic) private(thread_index, j)
    //#pragma omp parallel for schedule(guided) private(thread_index, j, total_index, a8, vb8, tmp8, tmp8_mul, jmax, jres, col_dim, mat_vec_i)
    #pragma omp parallel for schedule(static) private(thread_index, j, total_index, a8, vb8, tmp8, tmp8_mul, jmax, jres, col_dim, mat_vec_i)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//ret->element[i] = 0.0;
		//tmp8 = _mm512_setzero_pd();
		//_bncavx512_set0_dd(tmp8[thread_index]);
		_bncavx512_set0_dd(tmp8);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres = mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		//jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

        //total_index[thread_index] = 0;
        //for(j = 0; j < i; j++) total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;
        total_index = 0;
        for(j = 0; j < i; j++) total_index += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		//for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			/* a8[thread_index][0] = _mm512_load_pd(&(mat->element[0][total_index[thread_index]]));
			a8[thread_index][1] = _mm512_load_pd(&(mat->element[1][total_index[thread_index]]));
			vb8[thread_index][0] = _mm512_set_pd(
				vec->element[0][mat->nzero_index[i][j + 7]],
				vec->element[0][mat->nzero_index[i][j + 6]],
				vec->element[0][mat->nzero_index[i][j + 5]],
				vec->element[0][mat->nzero_index[i][j + 4]],
				vec->element[0][mat->nzero_index[i][j + 3]],
				vec->element[0][mat->nzero_index[i][j + 2]],
				vec->element[0][mat->nzero_index[i][j + 1]],
				vec->element[0][mat->nzero_index[i][j    ]]
			);
			vb8[thread_index][1] = _mm512_set_pd(
				vec->element[1][mat->nzero_index[i][j + 7]],
				vec->element[1][mat->nzero_index[i][j + 6]],
				vec->element[1][mat->nzero_index[i][j + 5]],
				vec->element[1][mat->nzero_index[i][j + 4]],
				vec->element[1][mat->nzero_index[i][j + 3]],
				vec->element[1][mat->nzero_index[i][j + 2]],
				vec->element[1][mat->nzero_index[i][j + 1]],
				vec->element[1][mat->nzero_index[i][j    ]]
			); */
			a8[0] = _mm512_load_pd(&(mat->element[0][total_index]));
			a8[1] = _mm512_load_pd(&(mat->element[1][total_index]));
			vb8[0] = _mm512_set_pd(
				vec->element[0][mat->nzero_index[i][j + 7]],
				vec->element[0][mat->nzero_index[i][j + 6]],
				vec->element[0][mat->nzero_index[i][j + 5]],
				vec->element[0][mat->nzero_index[i][j + 4]],
				vec->element[0][mat->nzero_index[i][j + 3]],
				vec->element[0][mat->nzero_index[i][j + 2]],
				vec->element[0][mat->nzero_index[i][j + 1]],
				vec->element[0][mat->nzero_index[i][j    ]]
			);
			vb8[1] = _mm512_set_pd(
				vec->element[1][mat->nzero_index[i][j + 7]],
				vec->element[1][mat->nzero_index[i][j + 6]],
				vec->element[1][mat->nzero_index[i][j + 5]],
				vec->element[1][mat->nzero_index[i][j + 4]],
				vec->element[1][mat->nzero_index[i][j + 3]],
				vec->element[1][mat->nzero_index[i][j + 2]],
				vec->element[1][mat->nzero_index[i][j + 1]],
				vec->element[1][mat->nzero_index[i][j    ]]
			);

			//tmp8 = _mm512_fmadd_pd(a4, vb4, tmp4);
			//_bncavx512_rdd_mul(tmp8_mul[thread_index], a8[thread_index], vb8[thread_index]);
			//_bncavx512_rdd_add(tmp8[thread_index], tmp8[thread_index], tmp8_mul[thread_index]);
			_bncavx512_rdd_mul(tmp8_mul, a8, vb8);
			_bncavx512_rdd_add(tmp8, tmp8, tmp8_mul);

			// total_index++;
			//total_index[thread_index] += _BNC_D_WIDTH;
			total_index += _BNC_D_WIDTH;
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];
		//_bncavx512_rdd_sum512d(mat_vec_i[thread_index], tmp8[thread_index]);
		_bncavx512_rdd_sum512d(mat_vec_i, tmp8);

		//set_ddvector_i(ret, i, mat_vec_i[thread_index]);
		set_ddvector_i(ret, i, mat_vec_i);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP DD SpMV: per-row parallel; inner loop uses VL-agnostic
	 * hardware gather + _bncsve2_rdd_mul/add multi-precision helpers. */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel for schedule(static) private(thread_index, j, total_index)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k;
			thread_index = omp_get_thread_num();
			total_index = 0;
			for(_k = 0; _k < i; _k++) total_index += mat->real_nzero_col_dim[_k];

			svfloat64_t t0, t1;
			double mat_vec_i_l[DDSIZE];
			_bncsve2_rdd_set0(&t0, &t1);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / _vl) * _vl;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += _vl)
			{
				svbool_t pg = svptrue_b64();
				svfloat64_t a0 = svld1_f64(pg, &(mat->element[0][total_index]));
				svfloat64_t a1 = svld1_f64(pg, &(mat->element[1][total_index]));
				svint64_t   idx= svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
				svfloat64_t b0 = svld1_gather_s64index_f64(pg, vec->element[0], idx);
				svfloat64_t b1 = svld1_gather_s64index_f64(pg, vec->element[1], idx);
				svfloat64_t m0, m1;
				_bncsve2_rdd_mul(pg, &m0, &m1, a0, a1, b0, b1);
				_bncsve2_rdd_add(pg, &t0, &t1, t0, t1, m0, m1);
				total_index += _vl;
			}
			mat_vec_i_l[0] = svaddv_f64(svptrue_b64(), t0);
			mat_vec_i_l[1] = svaddv_f64(svptrue_b64(), t1);
			{
				double mij[DDSIZE], vj[DDSIZE], pr[DDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
					mij[0] = mat->element[0][total_index];
					mij[1] = mat->element[1][total_index];
					vj[0]  = vec->element[0][idx_t];
					vj[1]  = vec->element[1][idx_t];
					rdd_mul(pr, mij, vj);
					rdd_add(mat_vec_i_l, mat_vec_i_l, pr);
					total_index++;
				}
			}
			set_ddvector_i(ret, i, mat_vec_i_l);
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel for schedule(static) private(thread_index, j, total_index)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k;
			thread_index = omp_get_thread_num();
			total_index = 0;
			for(_k = 0; _k < i; _k++) total_index += mat->real_nzero_col_dim[_k];

			float64x2_t a2[DDSIZE], vb2[DDSIZE], tmp2[DDSIZE], tmp4_mul[DDSIZE];
			double mat_vec_i_l[DDSIZE], vset[2];
			_bncneon_rdd_set0(tmp2);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / 2) * 2;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += 2)
			{
				a2[0] = vld1q_f64(&(mat->element[0][total_index]));
				a2[1] = vld1q_f64(&(mat->element[1][total_index]));
				vset[0] = vec->element[0][mat->nzero_index[i][j    ]];
				vset[1] = vec->element[0][mat->nzero_index[i][j + 1]];
				vb2[0] = vld1q_f64(vset);
				vset[0] = vec->element[1][mat->nzero_index[i][j    ]];
				vset[1] = vec->element[1][mat->nzero_index[i][j + 1]];
				vb2[1] = vld1q_f64(vset);
				_bncneon_rdd_mul(tmp4_mul, a2, vb2);
				_bncneon_rdd_add(tmp2, tmp2, tmp4_mul);
				total_index += 2;
			}
			_bncneon_rdd_sum128d(mat_vec_i_l, tmp2);
			{
				double mij[DDSIZE], vj[DDSIZE], pr[DDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
					mij[0] = mat->element[0][total_index];
					mij[1] = mat->element[1][total_index];
					vj[0]  = vec->element[0][idx_t];
					vj[1]  = vec->element[1][idx_t];
					rdd_mul(pr, mij, vj);
					rdd_add(mat_vec_i_l, mat_vec_i_l, pr);
					total_index++;
				}
			}
			set_ddvector_i(ret, i, mat_vec_i_l);
		}
	}

#else // others

    //#pragma omp parallel for private(thread_index, j)
	//#pragma omp parallel for schedule(guided) private(thread_index, j, total_index, mat_ij, vec_j, ret_i)
	//#pragma omp parallel for schedule(dynamic) private(thread_index, j, total_index, mat_ij, vec_j, ret_i)
	//#pragma omp parallel for schedule(static) private(thread_index, j, total_index, mat_ij, vec_j, ret_i) // threadprivate(thread_index, j, total_index, mat_ij, vec_j, ret_i)
	#pragma omp parallel for schedule(static) private(thread_index, j, total_index, mat_ij, vec_j, ret_i)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//get_mpfvector_i(ret, i) = 0.0;
		//mpf_set_ui(get_mpfvector_i(ret, i), 0UL);
		//rdd_set_ui(get_ddvector_i(ret, i), 0UL);
		//rdd_set_ui(ret_i[thread_index], 0UL);
		rdd_set_ui(ret_i, 0UL);

        //total_index[thread_index] = 0;
        total_index = 0;
       // for(j = 0; j < i; j++) total_index[thread_index] += mat->real_nzero_col_dim[j];
        for(j = 0; j < i; j++) total_index += mat->real_nzero_col_dim[j];

		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpf_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			//rdd_mul(tmp, (mpf_ptr)(mat->element[total_index]), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			/*
			mat_ij[thread_index][0] = mat->element[0][total_index[thread_index]];
			mat_ij[thread_index][1] = mat->element[1][total_index[thread_index]];
			vec_j[thread_index][0] = vec->element[0][mat->nzero_index[i][j]];
			vec_j[thread_index][1] = vec->element[1][mat->nzero_index[i][j]];
			rdd_mul(tmp[thread_index], mat_ij[thread_index], vec_j[thread_index]);
			//mpf_add(get_mpfvector_i(ret, i), get_mpfvector_i(ret, i), tmp);
			rdd_add(ret_i[thread_index], ret_i[thread_index], tmp[thread_index]);
			total_index[thread_index]++;
			*/
			mat_ij[0] = mat->element[0][total_index];
			mat_ij[1] = mat->element[1][total_index];
			vec_j[0] = vec->element[0][mat->nzero_index[i][j]];
			vec_j[1] = vec->element[1][mat->nzero_index[i][j]];
			rdd_mul(tmp, mat_ij, vec_j);
			//mpf_add(get_mpfvector_i(ret, i), get_mpfvector_i(ret, i), tmp);
			rdd_add(ret_i, ret_i, tmp);
			total_index++;
		}
		set_ddvector_i(ret, i, ret_i);
	}

#endif // __AVX2__

	return SUCCESS;
}

/* Multiply DRSMatrix * DDVector */
int _bncomp_mul_drsmatrix_ddvec(DDVector ret, DRSMatrix mat, DDVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	//ddfloat tmp, mat_ij, vec_j, ret_i;
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE], mat_ij[BNCOMP_MAX_NUM_THREADS], vec_j[BNCOMP_MAX_NUM_THREADS][DDSIZE], ret_i[BNCOMP_MAX_NUM_THREADS][DDSIZE];
    int thread_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS], vb4[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp4_mul[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double mat_vec_i[BNCOMP_MAX_NUM_THREADS][DDSIZE], vset[BNCOMP_MAX_NUM_THREADS][_BNC_D_WIDTH];

    //#pragma omp parallel for private(thread_index, j)
    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//ret->element[i] = 0.0;
		//tmp4 = _mm256_setzero_pd();
		_bncavx2_set0_dd(tmp4[thread_index]);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		//printf("%ld %ld jmax, jrec = %ld, %ld\n", i, total_index, jmax, jres);
		for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//printf("load0 ");fflush(stdout);
			a4[thread_index]  = _mm256_load_pd(&(mat->element[total_index[thread_index]]));
			//printf("load1 ");fflush(stdout);
			//a4[thread_index][1]  = _mm256_load_pd(&(mat->element[1][total_index[thread_index]]));

			//printf("set0 "); 
			// %ld, %ld, %ld, %ld ", mat->nzero_index[i][j+3], mat->nzero_index[i][j+2], mat->nzero_index[i][j+1], mat->nzero_index[i][j]); fflush(stdout)
			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[0][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[0][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[0][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[0][mat->nzero_index[i][j + 3]];
			//printf("set0 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][0] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[0][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[0][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[0][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[0][mat->nzero_index[i][j    ]]
			);

			//printf("set1 ");
			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[1][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[1][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[1][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[1][mat->nzero_index[i][j + 3]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][1] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);
			//printf("mul_add ");
			//tmp4 = _mm256_fmadd_pd(a4, vb4, tmp4);
			//_bncavx2_rdd_mul(tmp4_mul[thread_index], a4[thread_index], vb4[thread_index]);
			_bncavx2_rdd_mul_d(tmp4_mul[thread_index], vb4[thread_index], a4[thread_index]);
			_bncavx2_rdd_add(tmp4[thread_index], tmp4[thread_index], tmp4_mul[thread_index]);

			// total_index++;
			total_index[thread_index] += _BNC_D_WIDTH;
			//printf("%ld ", j);
		}
		//printf("\n");
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];
		_bncavx2_rdd_sum256d(mat_vec_i[thread_index], tmp4[thread_index]);

		//printf("%ld %ld mat_vec_i = %25.17e, %ld\n", i, total_index, mat_vec_i[0], mat->nzero_col_dim[i]);
		col_dim[thread_index] = mat->nzero_col_dim[i];
		set_ddvector_i(ret, i, mat_vec_i[thread_index]);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS], vb8[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp8_mul[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double mat_vec_i[BNCOMP_MAX_NUM_THREADS][DDSIZE];

    //#pragma omp parallel for private(thread_index, j)
    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//ret->element[i] = 0.0;
		//tmp8 = _mm512_setzero_pd();
		_bncavx512_set0_dd(tmp8[thread_index]);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres = mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a8[thread_index] = _mm512_load_pd(&(mat->element[total_index[thread_index]]));
			//a8[thread_index][1] = _mm512_load_pd(&(mat->element[1][total_index[thread_index]]));
			vb8[thread_index][0] = _mm512_set_pd(
				vec->element[0][mat->nzero_index[i][j + 7]],
				vec->element[0][mat->nzero_index[i][j + 6]],
				vec->element[0][mat->nzero_index[i][j + 5]],
				vec->element[0][mat->nzero_index[i][j + 4]],
				vec->element[0][mat->nzero_index[i][j + 3]],
				vec->element[0][mat->nzero_index[i][j + 2]],
				vec->element[0][mat->nzero_index[i][j + 1]],
				vec->element[0][mat->nzero_index[i][j    ]]
			);
			vb8[thread_index][1] = _mm512_set_pd(
				vec->element[1][mat->nzero_index[i][j + 7]],
				vec->element[1][mat->nzero_index[i][j + 6]],
				vec->element[1][mat->nzero_index[i][j + 5]],
				vec->element[1][mat->nzero_index[i][j + 4]],
				vec->element[1][mat->nzero_index[i][j + 3]],
				vec->element[1][mat->nzero_index[i][j + 2]],
				vec->element[1][mat->nzero_index[i][j + 1]],
				vec->element[1][mat->nzero_index[i][j    ]]
			);

			//tmp8 = _mm512_fmadd_pd(a4, vb4, tmp4);
			//_bncavx512_rdd_mul(tmp8_mul[thread_index], a8[thread_index], vb8[thread_index]);
			_bncavx512_rdd_mul_d(tmp8_mul[thread_index], vb8[thread_index], a8[thread_index]);
			_bncavx512_rdd_add(tmp8[thread_index], tmp8[thread_index], tmp8_mul[thread_index]);

			// total_index++;
			total_index[thread_index] += _BNC_D_WIDTH;
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];
		_bncavx512_rdd_sum512d(mat_vec_i[thread_index], tmp8[thread_index]);

		set_ddvector_i(ret, i, mat_vec_i[thread_index]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP D*DD SpMV: mat is plain double, vec is DD; gather DD limbs. */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel for schedule(static) private(thread_index, j)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k, _ti;
			thread_index = omp_get_thread_num();
			_ti = 0;
			for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];

			svfloat64_t t0, t1;
			double mat_vec_i_l[DDSIZE];
			_bncsve2_rdd_set0(&t0, &t1);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / _vl) * _vl;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += _vl)
			{
				svbool_t pg = svptrue_b64();
				svfloat64_t a_v = svld1_f64(pg, &(mat->element[_ti]));
				svint64_t   idx = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
				svfloat64_t b0  = svld1_gather_s64index_f64(pg, vec->element[0], idx);
				svfloat64_t b1  = svld1_gather_s64index_f64(pg, vec->element[1], idx);
				svfloat64_t m0, m1;
				_bncsve2_rdd_mul_d(pg, &m0, &m1, b0, b1, a_v);
				_bncsve2_rdd_add(pg, &t0, &t1, t0, t1, m0, m1);
				_ti += _vl;
			}
			mat_vec_i_l[0] = svaddv_f64(svptrue_b64(), t0);
			mat_vec_i_l[1] = svaddv_f64(svptrue_b64(), t1);
			{
				double vj[DDSIZE], pr[DDSIZE], mij;
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
					mij   = mat->element[_ti];
					vj[0] = vec->element[0][idx_t];
					vj[1] = vec->element[1][idx_t];
					rdd_mul_d(pr, vj, mij);
					rdd_add(mat_vec_i_l, mat_vec_i_l, pr);
					_ti++;
				}
			}
			set_ddvector_i(ret, i, mat_vec_i_l);
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel for schedule(static) private(thread_index, j)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k, _ti;
			thread_index = omp_get_thread_num();
			_ti = 0;
			for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];

			float64x2_t a2, vb2[DDSIZE], tmp2[DDSIZE], tmp4_mul[DDSIZE];
			double mat_vec_i_l[DDSIZE], vset[2];
			_bncneon_rdd_set0(tmp2);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / 2) * 2;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += 2)
			{
				a2 = vld1q_f64(&(mat->element[_ti]));
				vset[0] = vec->element[0][mat->nzero_index[i][j    ]];
				vset[1] = vec->element[0][mat->nzero_index[i][j + 1]];
				vb2[0] = vld1q_f64(vset);
				vset[0] = vec->element[1][mat->nzero_index[i][j    ]];
				vset[1] = vec->element[1][mat->nzero_index[i][j + 1]];
				vb2[1] = vld1q_f64(vset);
				_bncneon_rdd_mul_d(tmp4_mul, vb2, a2);
				_bncneon_rdd_add(tmp2, tmp2, tmp4_mul);
				_ti += 2;
			}
			_bncneon_rdd_sum128d(mat_vec_i_l, tmp2);
			{
				double vj[DDSIZE], pr[DDSIZE], mij;
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
					mij   = mat->element[_ti];
					vj[0] = vec->element[0][idx_t];
					vj[1] = vec->element[1][idx_t];
					rdd_mul_d(pr, vj, mij);
					rdd_add(mat_vec_i_l, mat_vec_i_l, pr);
					_ti++;
				}
			}
			set_ddvector_i(ret, i, mat_vec_i_l);
		}
	}

#else // others

    //#pragma omp parallel for private(thread_index, j)
    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//get_mpfvector_i(ret, i) = 0.0;
		//mpf_set_ui(get_mpfvector_i(ret, i), 0UL);
		//rdd_set_ui(get_ddvector_i(ret, i), 0UL);
		rdd_set_ui(ret_i[thread_index], 0UL);

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j];

		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpf_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			//rdd_mul(tmp, (mpf_ptr)(mat->element[total_index]), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mat_ij[thread_index] = mat->element[total_index[thread_index]];
			//mat_ij[thread_index][1] = mat->element[1][total_index[thread_index]];
			vec_j[thread_index][0] = vec->element[0][mat->nzero_index[i][j]];
			vec_j[thread_index][1] = vec->element[1][mat->nzero_index[i][j]];
			//rdd_mul(tmp[thread_index], mat_ij[thread_index], vec_j[thread_index]);
			rdd_mul_d(tmp[thread_index], vec_j[thread_index], mat_ij[thread_index]);
			//mpf_add(get_mpfvector_i(ret, i), get_mpfvector_i(ret, i), tmp);
			rdd_add(ret_i[thread_index], ret_i[thread_index], tmp[thread_index]);
			total_index[thread_index]++;
		}
		set_ddvector_i(ret, i, ret_i[thread_index]);
	}

#endif // __AVX2__

	return SUCCESS;
}

/* Multiply DDRSMatrix^T * DDVector */
int _bncomp_mul_ddrsmatrixt_ddvec(DDVector ret, DDRSMatrix mat, DDVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	//ddfloat tmp, mat_ji, vec_j, ret_j;
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE], mat_ji[BNCOMP_MAX_NUM_THREADS][DDSIZE], vec_i[BNCOMP_MAX_NUM_THREADS][DDSIZE], ret_j[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	DDVector work_vec[BNCOMP_MAX_NUM_THREADS];
    int thread_index, num_threads;
	long int div_row, start_i, end_i, row_dim;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}
    row_dim = mat->row_dim;

	//for(i = 0; i < mat->row_dim; i++)
	//	mpf_set_ui(get_mpfvector_i(ret, i), 0UL);

	// ret := 0
	set0_ddvector(ret);
	//total_index = 0;

	//#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
	#pragma omp parallel private(thread_index) // , num_threads) //, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		//num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_ddvector(vec->dim);
    }

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS][DDSIZE], vb4[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][DDSIZE], ret4[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double vset[BNCOMP_MAX_NUM_THREADS][_BNC_D_WIDTH];

	//#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)    
	{
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		//work_vec[thread_index] = init_ddvector(vec->dim);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		//for(i = 0; i < mat->row_dim; i++)
		for(i = start_i; i < end_i; i++)
		{
			thread_index = omp_get_thread_num();

			vb4[thread_index][0] = _mm256_set1_pd(vec->element[0][i]);
			vb4[thread_index][1] = _mm256_set1_pd(vec->element[1][i]);

			//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
			jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//jmax = (mat->row_dim / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->row_dim % _BNC_D_WIDTH;
			//#pragma omp ordered
			for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
			{
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				a4[thread_index][0] = _mm256_load_pd(&(mat->element[0][total_index[thread_index]]));
				a4[thread_index][1] = _mm256_load_pd(&(mat->element[1][total_index[thread_index]]));

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]];
				ret4[thread_index][0] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[0][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[0][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[0][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[0][mat->nzero_index[i][j    ]]
				);

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]];
				ret4[thread_index][1] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[1][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[1][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[1][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[1][mat->nzero_index[i][j    ]]
				);

				//ret4 = _mm256_fmadd_pd(a4, vb4, ret4);
				_bncavx2_rdd_mul(tmp4[thread_index], a4[thread_index], vb4[thread_index]);

			// #pragma omp critical
				_bncavx2_rdd_add(ret4[thread_index], ret4[thread_index], tmp4[thread_index]);

				if((j    ) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]] = ret4[thread_index][0][0];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]] = ret4[thread_index][1][0];
				}
				if((j + 1) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]] = ret4[thread_index][0][1];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]] = ret4[thread_index][1][1];
				}
				if((j + 2) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]] = ret4[thread_index][0][2];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]] = ret4[thread_index][1][2];
				}
				if((j + 3) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]] = ret4[thread_index][0][3];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]] = ret4[thread_index][1][3];
				}

				// total_index++;
				total_index[thread_index] += _BNC_D_WIDTH;
			}
			//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);

			//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		}
	} // #pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS][DDSIZE], vb8[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][DDSIZE], ret8[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];

	//#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		//work_vec[thread_index] = init_ddvector(vec->dim);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		//#pragma omp parallel for private(thread_index, j) ordered
		//for(i = 0; i < mat->row_dim; i++)
		for(i = start_i; i < end_i; i++)
		{
			thread_index = omp_get_thread_num();

			vb8[thread_index][0] = _mm512_set1_pd(vec->element[0][i]);
			vb8[thread_index][1] = _mm512_set1_pd(vec->element[1][i]);

			//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
			jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//#pragma omp ordered
			for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
			{
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				a8[thread_index][0] = _mm512_load_pd(&(mat->element[0][total_index[thread_index]]));
				a8[thread_index][1] = _mm512_load_pd(&(mat->element[1][total_index[thread_index]]));

				ret8[thread_index][0] = _mm512_set_pd(
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]]
				);
				ret8[thread_index][1] = _mm512_set_pd(
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]]
				);

				//ret8 = _mm512_fmadd_pd(a8, vb8, ret8);
				_bncavx512_rdd_mul(tmp8[thread_index], a8[thread_index], vb8[thread_index]);

				//#pragma omp critical
				_bncavx512_rdd_add(ret8[thread_index], ret8[thread_index], tmp8[thread_index]);

				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 7]] = ret8[thread_index][0][7];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 6]] = ret8[thread_index][0][6];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 5]] = ret8[thread_index][0][5];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 4]] = ret8[thread_index][0][4];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]] = ret8[thread_index][0][3];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]] = ret8[thread_index][0][2];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]] = ret8[thread_index][0][1];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]] = ret8[thread_index][0][0];

				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 7]] = ret8[thread_index][1][7];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 6]] = ret8[thread_index][1][6];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 5]] = ret8[thread_index][1][5];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 4]] = ret8[thread_index][1][4];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]] = ret8[thread_index][1][3];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]] = ret8[thread_index][1][2];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]] = ret8[thread_index][1][1];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]] = ret8[thread_index][1][0];

				// total_index++;
				total_index[thread_index] += _BNC_D_WIDTH;
			}
			//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);

			//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP DD A^T*x: each thread owns a private work_vec, so the
	 * gather/FMA/scatter on work_vec is race-free; results reduced into ret. */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;

			for(i = start_i; i < end_i; i++)
			{
				svfloat64_t vb0 = svdup_n_f64(vec->element[0][i]);
				svfloat64_t vb1 = svdup_n_f64(vec->element[1][i]);
				long int _ti = 0, _k;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / _vl) * _vl;
				long int jres = nz - jmax;
				double *w0 = work_vec[thread_index]->element[0];
				double *w1 = work_vec[thread_index]->element[1];
				for(j = 0; j < jmax; j += _vl)
				{
					svbool_t pg = svptrue_b64();
					svfloat64_t a0 = svld1_f64(pg, &(mat->element[0][_ti]));
					svfloat64_t a1 = svld1_f64(pg, &(mat->element[1][_ti]));
					svint64_t idx  = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
					svfloat64_t r0 = svld1_gather_s64index_f64(pg, w0, idx);
					svfloat64_t r1 = svld1_gather_s64index_f64(pg, w1, idx);
					svfloat64_t m0, m1;
					_bncsve2_rdd_mul(pg, &m0, &m1, a0, a1, vb0, vb1);
					_bncsve2_rdd_add(pg, &r0, &r1, r0, r1, m0, m1);
					svst1_scatter_s64index_f64(pg, w0, idx, r0);
					svst1_scatter_s64index_f64(pg, w1, idx, r1);
					_ti += _vl;
				}
				{
					double mij[DDSIZE], vi[DDSIZE], pr[DDSIZE], cur[DDSIZE];
					long int col_dim = mat->nzero_col_dim[i];
					long int _t;
					vi[0] = vec->element[0][i];
					vi[1] = vec->element[1][i];
					for(_t = 0; _t < jres; _t++)
					{
						long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
						mij[0] = mat->element[0][_ti];
						mij[1] = mat->element[1][_ti];
						cur[0] = w0[idx_t];
						cur[1] = w1[idx_t];
						rdd_mul(pr, mij, vi);
						rdd_add(cur, cur, pr);
						w0[idx_t] = cur[0];
						w1[idx_t] = cur[1];
						_ti++;
					}
				}
			}
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;

			for(i = start_i; i < end_i; i++)
			{
				float64x2_t vb2[DDSIZE], a2[DDSIZE], r2[DDSIZE], m2[DDSIZE];
				vb2[0] = vdupq_n_f64(vec->element[0][i]);
				vb2[1] = vdupq_n_f64(vec->element[1][i]);
				long int _ti = 0, _k;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / 2) * 2;
				long int jres = nz - jmax;
				double *w0 = work_vec[thread_index]->element[0];
				double *w1 = work_vec[thread_index]->element[1];
				double vset[2];
				for(j = 0; j < jmax; j += 2)
				{
					a2[0] = vld1q_f64(&(mat->element[0][_ti]));
					a2[1] = vld1q_f64(&(mat->element[1][_ti]));
					vset[0] = w0[mat->nzero_index[i][j]]; vset[1] = w0[mat->nzero_index[i][j + 1]];
					r2[0] = vld1q_f64(vset);
					vset[0] = w1[mat->nzero_index[i][j]]; vset[1] = w1[mat->nzero_index[i][j + 1]];
					r2[1] = vld1q_f64(vset);
					_bncneon_rdd_mul(m2, a2, vb2);
					_bncneon_rdd_add(r2, r2, m2);
					w0[mat->nzero_index[i][j    ]] = vgetq_lane_f64(r2[0], 0);
					w1[mat->nzero_index[i][j    ]] = vgetq_lane_f64(r2[1], 0);
					w0[mat->nzero_index[i][j + 1]] = vgetq_lane_f64(r2[0], 1);
					w1[mat->nzero_index[i][j + 1]] = vgetq_lane_f64(r2[1], 1);
					_ti += 2;
				}
				{
					double mij[DDSIZE], vi[DDSIZE], pr[DDSIZE], cur[DDSIZE];
					long int col_dim = mat->nzero_col_dim[i];
					long int _t;
					vi[0] = vec->element[0][i];
					vi[1] = vec->element[1][i];
					for(_t = 0; _t < jres; _t++)
					{
						long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
						mij[0] = mat->element[0][_ti];
						mij[1] = mat->element[1][_ti];
						cur[0] = w0[idx_t];
						cur[1] = w1[idx_t];
						rdd_mul(pr, mij, vi);
						rdd_add(cur, cur, pr);
						w0[idx_t] = cur[0];
						w1[idx_t] = cur[1];
						_ti++;
					}
				}
			}
		}
	}

#else // __AVX2__

    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < row_dim; i++)
		{
			//tmp_ret[0][i] = (double *)calloc(real_nzero_col_dim[i]);
			//tmp_ret[1][i] = (double *)calloc(real_nzero_col_dim[i]);

			thread_index = omp_get_thread_num();

			vec_i[thread_index][0] = vec->element[0][i];
			vec_i[thread_index][1] = vec->element[1][i];

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//#pragma omp ordered
			//ret_j[thread_index][0] = ret->element[0][i];
			//ret_j[thread_index][1] = ret->element[1][i];
			//rdd_set(ret_j, get_ddvector_i(ret, i));
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{

				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				mat_ji[thread_index][0] = mat->element[0][total_index[thread_index]];
				mat_ji[thread_index][1] = mat->element[1][total_index[thread_index]];
				//ret_j[thread_index][0] = ret->element[0][mat->nzero_index[i][j]];
				//ret_j[thread_index][1] = ret->element[1][mat->nzero_index[i][j]];
				ret_j[thread_index][0] = work_vec[thread_index]->element[0][mat->nzero_index[i][j]];
				ret_j[thread_index][1] = work_vec[thread_index]->element[1][mat->nzero_index[i][j]];				
				//rdd_mul(tmp, mat->element[total_index]), get_ddvector_i(vec, i));
				rdd_mul(tmp[thread_index], mat_ji[thread_index], vec_i[thread_index]);
				//rdd_add(ret->element[mat->nzero_index[i][j]], ret->element[mat->nzero_index[i][j]], tmp);

				//#pragma omp critical
				//{
				rdd_add(ret_j[thread_index], ret_j[thread_index], tmp[thread_index]);
				//set_ddvector_i(word, mat->nzero_index[i][j], ret_j[thread_index]);
				set_ddvector_i(work_vec[thread_index], mat->nzero_index[i][j], ret_j[thread_index]);
				//}
				total_index[thread_index]++;
			}
			//set_ddvector_i(ret, i, ret_j[thread_index]);
		}
	//} // #omp parallel private(thread_index, num_threads, i, j)

#endif // __AVX2__

    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
		#pragma omp critical
		{
			add_ddvector(ret, ret, work_vec[thread_index]);
			free_ddvector(work_vec[thread_index]);
		}
	}

	return SUCCESS;
}

/* Multiply DRSMatrix^T * DDVector */
int _bncomp_mul_drsmatrixt_ddvec(DDVector ret, DRSMatrix mat, DDVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	//ddfloat tmp, mat_ji, vec_j, ret_j;
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE], mat_ji[BNCOMP_MAX_NUM_THREADS], vec_i[BNCOMP_MAX_NUM_THREADS][DDSIZE], ret_j[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	DDVector work_vec[BNCOMP_MAX_NUM_THREADS];
    int thread_index, num_threads;
	long int div_row, start_i, end_i;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//for(i = 0; i < mat->row_dim; i++)
	//	mpf_set_ui(get_mpfvector_i(ret, i), 0UL);

	// ret := 0
	set0_ddvector(ret);
	//total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS], vb4[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][DDSIZE], ret4[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double vset[BNCOMP_MAX_NUM_THREADS][_BNC_D_WIDTH];

	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_ddvector(vec->dim);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		//for(i = 0; i < mat->row_dim; i++)
		for(i = start_i; i < end_i; i++)
		{
			thread_index = omp_get_thread_num();

			vb4[thread_index][0] = _mm256_set1_pd(vec->element[0][i]);
			vb4[thread_index][1] = _mm256_set1_pd(vec->element[1][i]);

			//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
			jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//jmax = (mat->row_dim / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->row_dim % _BNC_D_WIDTH;
			//#pragma omp ordered
			for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
			{
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				a4[thread_index] = _mm256_load_pd(&(mat->element[total_index[thread_index]]));
				//a4[thread_index][1] = _mm256_load_pd(&(mat->element[1][total_index[thread_index]]));

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]];
				ret4[thread_index][0] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[0][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[0][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[0][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[0][mat->nzero_index[i][j    ]]
				);

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]];
				ret4[thread_index][1] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[1][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[1][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[1][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[1][mat->nzero_index[i][j    ]]
				);

				//ret4 = _mm256_fmadd_pd(a4, vb4, ret4);
				//_bncavx2_rdd_mul(tmp4[thread_index], a4[thread_index], vb4[thread_index]);
				_bncavx2_rdd_mul_d(tmp4[thread_index], vb4[thread_index], a4[thread_index]);

			// #pragma omp critical
				_bncavx2_rdd_add(ret4[thread_index], ret4[thread_index], tmp4[thread_index]);

				if((j    ) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]] = ret4[thread_index][0][0];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]] = ret4[thread_index][1][0];
				}
				if((j + 1) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]] = ret4[thread_index][0][1];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]] = ret4[thread_index][1][1];
				}
				if((j + 2) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]] = ret4[thread_index][0][2];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]] = ret4[thread_index][1][2];
				}
				if((j + 3) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]] = ret4[thread_index][0][3];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]] = ret4[thread_index][1][3];
				}

				// total_index++;
				total_index[thread_index] += _BNC_D_WIDTH;
			}
			//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);

			//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		}
	} // #pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS], vb8[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][DDSIZE], ret8[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_ddvector(vec->dim);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		//#pragma omp parallel for private(thread_index, j) ordered
		//for(i = 0; i < mat->row_dim; i++)
		for(i = start_i; i < end_i; i++)
		{
			thread_index = omp_get_thread_num();

			vb8[thread_index][0] = _mm512_set1_pd(vec->element[0][i]);
			vb8[thread_index][1] = _mm512_set1_pd(vec->element[1][i]);

			//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
			jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//#pragma omp ordered
			for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
			{
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				a8[thread_index] = _mm512_load_pd(&(mat->element[total_index[thread_index]]));
				//a8[thread_index][1] = _mm512_load_pd(&(mat->element[1][total_index[thread_index]]));

				ret8[thread_index][0] = _mm512_set_pd(
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]]
				);
				ret8[thread_index][1] = _mm512_set_pd(
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]]
				);

				//ret8 = _mm512_fmadd_pd(a8, vb8, ret8);
				//_bncavx512_rdd_mul(tmp8[thread_index], a8[thread_index], vb8[thread_index]);
				_bncavx512_rdd_mul_d(tmp8[thread_index], vb8[thread_index], a8[thread_index]);

				//#pragma omp critical
				_bncavx512_rdd_add(ret8[thread_index], ret8[thread_index], tmp8[thread_index]);

				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 7]] = ret8[thread_index][0][7];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 6]] = ret8[thread_index][0][6];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 5]] = ret8[thread_index][0][5];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 4]] = ret8[thread_index][0][4];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]] = ret8[thread_index][0][3];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]] = ret8[thread_index][0][2];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]] = ret8[thread_index][0][1];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]] = ret8[thread_index][0][0];

				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 7]] = ret8[thread_index][1][7];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 6]] = ret8[thread_index][1][6];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 5]] = ret8[thread_index][1][5];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 4]] = ret8[thread_index][1][4];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]] = ret8[thread_index][1][3];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]] = ret8[thread_index][1][2];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]] = ret8[thread_index][1][1];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]] = ret8[thread_index][1][0];

				// total_index++;
				total_index[thread_index] += _BNC_D_WIDTH;
			}
			//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);

			//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP D^T*DD: mat plain double, vec DD; per-thread work_vec. */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			work_vec[thread_index] = init_ddvector(vec->dim);
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;

			for(i = start_i; i < end_i; i++)
			{
				svfloat64_t vb0 = svdup_n_f64(vec->element[0][i]);
				svfloat64_t vb1 = svdup_n_f64(vec->element[1][i]);
				long int _ti = 0, _k;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / _vl) * _vl;
				long int jres = nz - jmax;
				double *w0 = work_vec[thread_index]->element[0];
				double *w1 = work_vec[thread_index]->element[1];
				for(j = 0; j < jmax; j += _vl)
				{
					svbool_t pg = svptrue_b64();
					svfloat64_t a_v = svld1_f64(pg, &(mat->element[_ti]));
					svint64_t idx  = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
					svfloat64_t r0 = svld1_gather_s64index_f64(pg, w0, idx);
					svfloat64_t r1 = svld1_gather_s64index_f64(pg, w1, idx);
					svfloat64_t m0, m1;
					_bncsve2_rdd_mul_d(pg, &m0, &m1, vb0, vb1, a_v);
					_bncsve2_rdd_add(pg, &r0, &r1, r0, r1, m0, m1);
					svst1_scatter_s64index_f64(pg, w0, idx, r0);
					svst1_scatter_s64index_f64(pg, w1, idx, r1);
					_ti += _vl;
				}
				{
					double vi[DDSIZE], pr[DDSIZE], cur[DDSIZE], aij;
					long int col_dim = mat->nzero_col_dim[i];
					long int _t;
					vi[0] = vec->element[0][i];
					vi[1] = vec->element[1][i];
					for(_t = 0; _t < jres; _t++)
					{
						long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
						aij = mat->element[_ti];
						cur[0] = w0[idx_t];
						cur[1] = w1[idx_t];
						rdd_mul_d(pr, vi, aij);
						rdd_add(cur, cur, pr);
						w0[idx_t] = cur[0];
						w1[idx_t] = cur[1];
						_ti++;
					}
				}
			}
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			work_vec[thread_index] = init_ddvector(vec->dim);
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;

			for(i = start_i; i < end_i; i++)
			{
				float64x2_t vb2[DDSIZE], r2[DDSIZE], m2[DDSIZE], a2;
				vb2[0] = vdupq_n_f64(vec->element[0][i]);
				vb2[1] = vdupq_n_f64(vec->element[1][i]);
				long int _ti = 0, _k;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / 2) * 2;
				long int jres = nz - jmax;
				double *w0 = work_vec[thread_index]->element[0];
				double *w1 = work_vec[thread_index]->element[1];
				double vset[2];
				for(j = 0; j < jmax; j += 2)
				{
					a2 = vld1q_f64(&(mat->element[_ti]));
					vset[0] = w0[mat->nzero_index[i][j]]; vset[1] = w0[mat->nzero_index[i][j + 1]];
					r2[0] = vld1q_f64(vset);
					vset[0] = w1[mat->nzero_index[i][j]]; vset[1] = w1[mat->nzero_index[i][j + 1]];
					r2[1] = vld1q_f64(vset);
					_bncneon_rdd_mul_d(m2, vb2, a2);
					_bncneon_rdd_add(r2, r2, m2);
					w0[mat->nzero_index[i][j    ]] = vgetq_lane_f64(r2[0], 0);
					w1[mat->nzero_index[i][j    ]] = vgetq_lane_f64(r2[1], 0);
					w0[mat->nzero_index[i][j + 1]] = vgetq_lane_f64(r2[0], 1);
					w1[mat->nzero_index[i][j + 1]] = vgetq_lane_f64(r2[1], 1);
					_ti += 2;
				}
				{
					double vi[DDSIZE], pr[DDSIZE], cur[DDSIZE], aij;
					long int col_dim = mat->nzero_col_dim[i];
					long int _t;
					vi[0] = vec->element[0][i];
					vi[1] = vec->element[1][i];
					for(_t = 0; _t < jres; _t++)
					{
						long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
						aij = mat->element[_ti];
						cur[0] = w0[idx_t];
						cur[1] = w1[idx_t];
						rdd_mul_d(pr, vi, aij);
						rdd_add(cur, cur, pr);
						w0[idx_t] = cur[0];
						w1[idx_t] = cur[1];
						_ti++;
					}
				}
			}
		}
	}

#else // __AVX2__
	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_ddvector(vec->dim);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		for(i = start_i; i < end_i; i++)
		{
			//tmp_ret[0][i] = (double *)calloc(real_nzero_col_dim[i]);
			//tmp_ret[1][i] = (double *)calloc(real_nzero_col_dim[i]);

			thread_index = omp_get_thread_num();

			vec_i[thread_index][0] = vec->element[0][i];
			vec_i[thread_index][1] = vec->element[1][i];

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//#pragma omp ordered
			//ret_j[thread_index][0] = ret->element[0][i];
			//ret_j[thread_index][1] = ret->element[1][i];
			//rdd_set(ret_j, get_ddvector_i(ret, i));
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{

				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				mat_ji[thread_index] = mat->element[total_index[thread_index]];
				//mat_ji[thread_index][1] = mat->element[1][total_index[thread_index]];
				//ret_j[thread_index][0] = ret->element[0][mat->nzero_index[i][j]];
				//ret_j[thread_index][1] = ret->element[1][mat->nzero_index[i][j]];
				ret_j[thread_index][0] = work_vec[thread_index]->element[0][mat->nzero_index[i][j]];
				ret_j[thread_index][1] = work_vec[thread_index]->element[1][mat->nzero_index[i][j]];				
				//rdd_mul(tmp, mat->element[total_index]), get_ddvector_i(vec, i));
				//rdd_mul(tmp[thread_index], mat_ji[thread_index], vec_i[thread_index]);
				rdd_mul_d(tmp[thread_index], vec_i[thread_index], mat_ji[thread_index]);
				//rdd_add(ret->element[mat->nzero_index[i][j]], ret->element[mat->nzero_index[i][j]], tmp);

				//#pragma omp critical
				//{
				rdd_add(ret_j[thread_index], ret_j[thread_index], tmp[thread_index]);
				//set_ddvector_i(word, mat->nzero_index[i][j], ret_j[thread_index]);
				set_ddvector_i(work_vec[thread_index], mat->nzero_index[i][j], ret_j[thread_index]);
				//}
				total_index[thread_index]++;
			}
			//set_ddvector_i(ret, i, ret_j[thread_index]);
		}
	} // #omp parallel private(thread_index, num_threads, i, j)

#endif // __AVX2__

    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
		#pragma omp critical
		{
			add_ddvector(ret, ret, work_vec[thread_index]);
			free_ddvector(work_vec[thread_index]);
		}
	}

	return SUCCESS;
}


/* Multiply CDDRSMatrix * CDDVector */
int _bncomp_mul_cddrsmatrix_cddvec(CDDVector ret, CDDRSMatrix mat, CDDVector vec)
{
	long int i, j, total_index;
	DDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_ddvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_ddrsmatrix_ddvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_ddrsmatrix_ddvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_ddrsmatrix_ddvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_ddrsmatrix_ddvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_sub_ddvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_ddvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_ddvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CDRSMatrix * CDDVector */
int _bncomp_mul_cdrsmatrix_cddvec(CDDVector ret, CDRSMatrix mat, CDDVector vec)
{
	long int i, j, total_index;
	DDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_ddvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_drsmatrix_ddvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_drsmatrix_ddvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_drsmatrix_ddvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_drsmatrix_ddvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_sub_ddvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_ddvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_ddvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CDDRSMatrix^T * CDDVector */
int _bncomp_mul_cddrsmatrixt_cddvec(CDDVector ret, CDDRSMatrix mat, CDDVector vec)
{
	long int i, j, total_index;
	DDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_ddvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_ddrsmatrixt_ddvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_ddrsmatrixt_ddvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_ddrsmatrixt_ddvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_ddrsmatrixt_ddvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_sub_ddvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_ddvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_ddvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CDRSMatrix^T * CDDVector */
int _bncomp_mul_cdrsmatrixt_cddvec(CDDVector ret, CDRSMatrix mat, CDDVector vec)
{
	long int i, j, total_index;
	DDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_ddvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_drsmatrixt_ddvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_drsmatrixt_ddvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_drsmatrixt_ddvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_drsmatrixt_ddvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_sub_ddvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_ddvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_ddvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply conj(CDDRSMatrix)^T * CDDVector */
int _bncomp_mul_cddrsmatrixs_cddvec(CDDVector ret, CDDRSMatrix mat, CDDVector vec)
{
	long int i, j, total_index;
	DDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_ddvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_ddrsmatrixt_ddvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_ddrsmatrixt_ddvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_ddrsmatrixt_ddvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_ddrsmatrixt_ddvec(tmp_vec[3], mat->im, vec->im);

	// mat_re * vec_re + mat_im * vec_im
	_bncomp_add_ddvector(ret->re, tmp_vec[0], tmp_vec[3]);
	// mat_re * vec_im - mat_im * vec_re
	_bncomp_sub_ddvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_ddvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply conj(CDRSMatrix)^T * CDDVector */
int _bncomp_mul_cdrsmatrixs_cddvec(CDDVector ret, CDRSMatrix mat, CDDVector vec)
{
	long int i, j, total_index;
	DDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_ddvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_drsmatrixt_ddvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_drsmatrixt_ddvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_drsmatrixt_ddvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_drsmatrixt_ddvec(tmp_vec[3], mat->im, vec->im);
 
	// mat_re * vec_re + mat_im * vec_im
	_bncomp_add_ddvector(ret->re, tmp_vec[0], tmp_vec[3]);
	// mat_re * vec_im - mat_im * vec_re
	_bncomp_sub_ddvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_ddvector(tmp_vec[i]);

	return SUCCESS;
}

//-----------------------------------------------
// TD precision
//-----------------------------------------------

/* Multiply TDRSMatrix * TDVector */
int _bncomp_mul_tdrsmatrix_tdvec(TDVector ret, TDRSMatrix mat, TDVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	//tdfloat tmp, mat_ij, vec_j, ret_i;
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], mat_ij[BNCOMP_MAX_NUM_THREADS][TDSIZE], vec_j[BNCOMP_MAX_NUM_THREADS][TDSIZE], ret_i[BNCOMP_MAX_NUM_THREADS][TDSIZE];
    int thread_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS][TDSIZE], vb4[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp4_mul[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double mat_vec_i[BNCOMP_MAX_NUM_THREADS][TDSIZE], vset[BNCOMP_MAX_NUM_THREADS][_BNC_D_WIDTH];

    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//ret->element[i] = 0.0;
		//tmp4 = _mm256_setzero_pd();
		_bncavx2_set0_td(tmp4[thread_index]);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;
		//printf("%ld jmax, jres = %ld, %ld\n", i, jmax, jres);

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
		{
			//printf("load "); fflush(stdout);
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a4[thread_index][0]  = _mm256_load_pd(&(mat->element[0][total_index[thread_index]]));
			a4[thread_index][1]  = _mm256_load_pd(&(mat->element[1][total_index[thread_index]]));
			a4[thread_index][2]  = _mm256_load_pd(&(mat->element[2][total_index[thread_index]]));

			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[0][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[0][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[0][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[0][mat->nzero_index[i][j + 3]];
			//printf("set0 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][0] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[0][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[0][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[0][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[0][mat->nzero_index[i][j    ]]
			);

			//printf("set1 ");
            vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[1][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[1][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[1][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[1][mat->nzero_index[i][j + 3]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][1] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);

			//printf("set2 ");
			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[2][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[2][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[2][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[2][mat->nzero_index[i][j + 3]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][2] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);

			//printf("mul_add "); fflush(stdout);
			//tmp4 = _mm256_fmadd_pd(a4, vb4, tmp4);
			_bncavx2_rtd_mul(tmp4_mul[thread_index], a4[thread_index], vb4[thread_index]);
			_bncavx2_rtd_add(tmp4[thread_index], tmp4[thread_index], tmp4_mul[thread_index]);

			// total_index++;
			total_index[thread_index] += _BNC_D_WIDTH;
			//printf("%ld ", j);
		}
		//total_index += mat->real_nzero_col_dim[i];
		//printf("\n");
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];
		_bncavx2_rtd_sum256d(mat_vec_i[thread_index], tmp4[thread_index]);

		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i[0], mat->nzero_col_dim[i], jmax, jres);

		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i[0]);
		set_tdvector_i(ret, i, mat_vec_i[thread_index]);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS][TDSIZE], vb8[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp8_mul[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double mat_vec_i[BNCOMP_MAX_NUM_THREADS][DDSIZE];

    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

        //ret->element[i] = 0.0;]
		//tmp8 = _mm512_setzero_pd();
		_bncavx512_set0_td(tmp8[thread_index]);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a8[thread_index][0] = _mm512_load_pd(&(mat->element[0][total_index[thread_index]]));
			a8[thread_index][1] = _mm512_load_pd(&(mat->element[1][total_index[thread_index]]));
			a8[thread_index][2] = _mm512_load_pd(&(mat->element[2][total_index[thread_index]]));

			vb8[thread_index][0] = _mm512_set_pd(
				vec->element[0][mat->nzero_index[i][j + 7]],
				vec->element[0][mat->nzero_index[i][j + 6]],
				vec->element[0][mat->nzero_index[i][j + 5]],
				vec->element[0][mat->nzero_index[i][j + 4]],
				vec->element[0][mat->nzero_index[i][j + 3]],
				vec->element[0][mat->nzero_index[i][j + 2]],
				vec->element[0][mat->nzero_index[i][j + 1]],
				vec->element[0][mat->nzero_index[i][j    ]]
			);
			vb8[thread_index][1] = _mm512_set_pd(
				vec->element[1][mat->nzero_index[i][j + 7]],
				vec->element[1][mat->nzero_index[i][j + 6]],
				vec->element[1][mat->nzero_index[i][j + 5]],
				vec->element[1][mat->nzero_index[i][j + 4]],
				vec->element[1][mat->nzero_index[i][j + 3]],
				vec->element[1][mat->nzero_index[i][j + 2]],
				vec->element[1][mat->nzero_index[i][j + 1]],
				vec->element[1][mat->nzero_index[i][j    ]]
			);
			vb8[thread_index][2] = _mm512_set_pd(
				vec->element[2][mat->nzero_index[i][j + 7]],
				vec->element[2][mat->nzero_index[i][j + 6]],
				vec->element[2][mat->nzero_index[i][j + 5]],
				vec->element[2][mat->nzero_index[i][j + 4]],
				vec->element[2][mat->nzero_index[i][j + 3]],
				vec->element[2][mat->nzero_index[i][j + 2]],
				vec->element[2][mat->nzero_index[i][j + 1]],
				vec->element[2][mat->nzero_index[i][j    ]]
			);

			//tmp8 = _mm512_fmadd_pd(a4, vb4, tmp4);
			_bncavx512_rtd_mul(tmp8_mul[thread_index], a8[thread_index], vb8[thread_index]);
			_bncavx512_rtd_add(tmp8[thread_index], tmp8[thread_index], tmp8_mul[thread_index]);

			// total_index++;
			total_index[thread_index] += _BNC_D_WIDTH;
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];
		_bncavx512_rtd_sum512d(mat_vec_i[thread_index], tmp8[thread_index]);

		set_tdvector_i(ret, i, mat_vec_i[thread_index]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP TD SpMV. */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel for schedule(static) private(thread_index, j)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k, _ti;
			thread_index = omp_get_thread_num();
			_ti = 0;
			for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];

			svfloat64_t t0, t1, t2;
			double mat_vec_i_l[TDSIZE];
			_bncsve2_rtd_set0(&t0, &t1, &t2);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / _vl) * _vl;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += _vl)
			{
				svbool_t pg = svptrue_b64();
				svfloat64_t a0 = svld1_f64(pg, &(mat->element[0][_ti]));
				svfloat64_t a1 = svld1_f64(pg, &(mat->element[1][_ti]));
				svfloat64_t a2v= svld1_f64(pg, &(mat->element[2][_ti]));
				svint64_t   idx= svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
				svfloat64_t b0 = svld1_gather_s64index_f64(pg, vec->element[0], idx);
				svfloat64_t b1 = svld1_gather_s64index_f64(pg, vec->element[1], idx);
				svfloat64_t b2 = svld1_gather_s64index_f64(pg, vec->element[2], idx);
				svfloat64_t m0, m1, m2;
				_bncsve2_rtd_mul(pg, &m0, &m1, &m2, a0, a1, a2v, b0, b1, b2);
				_bncsve2_rtd_add(pg, &t0, &t1, &t2, t0, t1, t2, m0, m1, m2);
				_ti += _vl;
			}
			mat_vec_i_l[0] = svaddv_f64(svptrue_b64(), t0);
			mat_vec_i_l[1] = svaddv_f64(svptrue_b64(), t1);
			mat_vec_i_l[2] = svaddv_f64(svptrue_b64(), t2);
			{
				double mij[TDSIZE], vj[TDSIZE], pr[TDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
					mij[0] = mat->element[0][_ti];
					mij[1] = mat->element[1][_ti];
					mij[2] = mat->element[2][_ti];
					vj[0]  = vec->element[0][idx_t];
					vj[1]  = vec->element[1][idx_t];
					vj[2]  = vec->element[2][idx_t];
					rtd_mul(pr, mij, vj);
					rtd_add(mat_vec_i_l, mat_vec_i_l, pr);
					_ti++;
				}
			}
			set_tdvector_i(ret, i, mat_vec_i_l);
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel for schedule(static) private(thread_index, j)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k, _ti;
			thread_index = omp_get_thread_num();
			_ti = 0;
			for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];

			float64x2_t a2[TDSIZE], vb2[TDSIZE], tmp2[TDSIZE], tmp4_mul[TDSIZE];
			double mat_vec_i_l[TDSIZE], vset[2];
			_bncneon_set0_td(tmp2);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / 2) * 2;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += 2)
			{
				a2[0] = vld1q_f64(&(mat->element[0][_ti]));
				a2[1] = vld1q_f64(&(mat->element[1][_ti]));
				a2[2] = vld1q_f64(&(mat->element[2][_ti]));
				vset[0] = vec->element[0][mat->nzero_index[i][j    ]];
				vset[1] = vec->element[0][mat->nzero_index[i][j + 1]];
				vb2[0] = vld1q_f64(vset);
				vset[0] = vec->element[1][mat->nzero_index[i][j    ]];
				vset[1] = vec->element[1][mat->nzero_index[i][j + 1]];
				vb2[1] = vld1q_f64(vset);
				vset[0] = vec->element[2][mat->nzero_index[i][j    ]];
				vset[1] = vec->element[2][mat->nzero_index[i][j + 1]];
				vb2[2] = vld1q_f64(vset);
				_bncneon_rtd_mul(tmp4_mul, a2, vb2);
				_bncneon_rtd_add(tmp2, tmp2, tmp4_mul);
				_ti += 2;
			}
			_bncneon_rtd_sum128d(mat_vec_i_l, tmp2);
			{
				double mij[TDSIZE], vj[TDSIZE], pr[TDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
					mij[0] = mat->element[0][_ti];
					mij[1] = mat->element[1][_ti];
					mij[2] = mat->element[2][_ti];
					vj[0]  = vec->element[0][idx_t];
					vj[1]  = vec->element[1][idx_t];
					vj[2]  = vec->element[2][idx_t];
					rtd_mul(pr, mij, vj);
					rtd_add(mat_vec_i_l, mat_vec_i_l, pr);
					_ti++;
				}
			}
			set_tdvector_i(ret, i, mat_vec_i_l);
		}
	}

#else // others

    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//get_mpfvector_i(ret, i) = 0.0;
		//mpf_set_ui(get_mpfvector_i(ret, i), 0UL);
		//rtd_set_ui(get_tdvector_i(ret, i), 0UL);
		rtd_set_ui(ret_i[thread_index], 0UL);

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpf_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			//rtd_mul(tmp, (mpf_ptr)(mat->element[total_index]), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mat_ij[thread_index][0] = mat->element[0][total_index[thread_index]];
			mat_ij[thread_index][1] = mat->element[1][total_index[thread_index]];
			mat_ij[thread_index][2] = mat->element[2][total_index[thread_index]];

			vec_j[thread_index][0] = vec->element[0][mat->nzero_index[i][j]];
			vec_j[thread_index][1] = vec->element[1][mat->nzero_index[i][j]];
			vec_j[thread_index][2] = vec->element[2][mat->nzero_index[i][j]];

			//printf("%d, %d: rtd_mul ", i, j);
			rtd_mul(tmp[thread_index], mat_ij[thread_index], vec_j[thread_index]);
			//mpf_add(get_mpfvector_i(ret, i), get_mpfvector_i(ret, i), tmp);
			//printf("rtd_add ");
			rtd_add(ret_i[thread_index], ret_i[thread_index], tmp[thread_index]);
			total_index[thread_index]++;
		}
		set_tdvector_i(ret, i, ret_i[thread_index]);
		//printf("set_tdvector_i %ld\n", i);
	}

#endif // __AVX2__

	return SUCCESS;
}

/* Multiply DRSMatrix * TDVector */
int _bncomp_mul_drsmatrix_tdvec(TDVector ret, DRSMatrix mat, TDVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	//tdfloat tmp, mat_ij, vec_j, ret_i;
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], mat_ij[BNCOMP_MAX_NUM_THREADS], vec_j[BNCOMP_MAX_NUM_THREADS][TDSIZE], ret_i[BNCOMP_MAX_NUM_THREADS][TDSIZE];
    int thread_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS], vb4[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp4_mul[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double mat_vec_i[BNCOMP_MAX_NUM_THREADS][TDSIZE], vset[BNCOMP_MAX_NUM_THREADS][_BNC_D_WIDTH];

    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//ret->element[i] = 0.0;
		//tmp4 = _mm256_setzero_pd();
		_bncavx2_set0_td(tmp4[thread_index]);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;
		//printf("%ld jmax, jres = %ld, %ld\n", i, jmax, jres);

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
		{
			//printf("load "); fflush(stdout);
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a4[thread_index]  = _mm256_load_pd(&(mat->element[total_index[thread_index]]));
			//a4[thread_index][1]  = _mm256_load_pd(&(mat->element[1][total_index[thread_index]]));
			//a4[thread_index][2]  = _mm256_load_pd(&(mat->element[2][total_index[thread_index]]));

			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[0][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[0][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[0][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[0][mat->nzero_index[i][j + 3]];
			//printf("set0 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][0] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[0][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[0][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[0][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[0][mat->nzero_index[i][j    ]]
			);

			//printf("set1 ");
            vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[1][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[1][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[1][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[1][mat->nzero_index[i][j + 3]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][1] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);

			//printf("set2 ");
			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[2][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[2][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[2][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[2][mat->nzero_index[i][j + 3]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][2] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);

			//printf("mul_add "); fflush(stdout);
			//tmp4 = _mm256_fmadd_pd(a4, vb4, tmp4);
			//_bncavx2_rtd_mul(tmp4_mul[thread_index], a4[thread_index], vb4[thread_index]);
			_bncavx2_rtd_mulq_d(tmp4_mul[thread_index], vb4[thread_index], a4[thread_index]);
			_bncavx2_rtd_add(tmp4[thread_index], tmp4[thread_index], tmp4_mul[thread_index]);

			// total_index++;
			total_index[thread_index] += _BNC_D_WIDTH;
			//printf("%ld ", j);
		}
		//total_index += mat->real_nzero_col_dim[i];
		//printf("\n");
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];
		_bncavx2_rtd_sum256d(mat_vec_i[thread_index], tmp4[thread_index]);

		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i[0], mat->nzero_col_dim[i], jmax, jres);

		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i[0]);
		set_tdvector_i(ret, i, mat_vec_i[thread_index]);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS], vb8[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp8_mul[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double mat_vec_i[BNCOMP_MAX_NUM_THREADS][DDSIZE];

    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

        //ret->element[i] = 0.0;]
		//tmp8 = _mm512_setzero_pd();
		_bncavx512_set0_td(tmp8[thread_index]);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a8[thread_index] = _mm512_load_pd(&(mat->element[total_index[thread_index]]));
			//a8[thread_index][1] = _mm512_load_pd(&(mat->element[1][total_index[thread_index]]));
			//a8[thread_index][2] = _mm512_load_pd(&(mat->element[2][total_index[thread_index]]));

			vb8[thread_index][0] = _mm512_set_pd(
				vec->element[0][mat->nzero_index[i][j + 7]],
				vec->element[0][mat->nzero_index[i][j + 6]],
				vec->element[0][mat->nzero_index[i][j + 5]],
				vec->element[0][mat->nzero_index[i][j + 4]],
				vec->element[0][mat->nzero_index[i][j + 3]],
				vec->element[0][mat->nzero_index[i][j + 2]],
				vec->element[0][mat->nzero_index[i][j + 1]],
				vec->element[0][mat->nzero_index[i][j    ]]
			);
			vb8[thread_index][1] = _mm512_set_pd(
				vec->element[1][mat->nzero_index[i][j + 7]],
				vec->element[1][mat->nzero_index[i][j + 6]],
				vec->element[1][mat->nzero_index[i][j + 5]],
				vec->element[1][mat->nzero_index[i][j + 4]],
				vec->element[1][mat->nzero_index[i][j + 3]],
				vec->element[1][mat->nzero_index[i][j + 2]],
				vec->element[1][mat->nzero_index[i][j + 1]],
				vec->element[1][mat->nzero_index[i][j    ]]
			);
			vb8[thread_index][2] = _mm512_set_pd(
				vec->element[2][mat->nzero_index[i][j + 7]],
				vec->element[2][mat->nzero_index[i][j + 6]],
				vec->element[2][mat->nzero_index[i][j + 5]],
				vec->element[2][mat->nzero_index[i][j + 4]],
				vec->element[2][mat->nzero_index[i][j + 3]],
				vec->element[2][mat->nzero_index[i][j + 2]],
				vec->element[2][mat->nzero_index[i][j + 1]],
				vec->element[2][mat->nzero_index[i][j    ]]
			);

			//tmp8 = _mm512_fmadd_pd(a4, vb4, tmp4);
			//_bncavx512_rtd_mul(tmp8_mul[thread_index], a8[thread_index], vb8[thread_index]);
			_bncavx512_rtd_mul_d(tmp8_mul[thread_index], vb8[thread_index], a8[thread_index]);
			//_bncavx512_rtd_mul_d(tmp8_mul[thread_index], a8[thread_index], vb8[thread_index]);
			_bncavx512_rtd_add(tmp8[thread_index], tmp8[thread_index], tmp8_mul[thread_index]);

			// total_index++;
			total_index[thread_index] += _BNC_D_WIDTH;
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];
		_bncavx512_rtd_sum512d(mat_vec_i[thread_index], tmp8[thread_index]);

		set_tdvector_i(ret, i, mat_vec_i[thread_index]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP D*TD SpMV: promote plain-double mat to TD (a,0,0). */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel for schedule(static) private(thread_index, j)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k, _ti;
			thread_index = omp_get_thread_num();
			_ti = 0;
			for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];

			svfloat64_t t0, t1, t2;
			double mat_vec_i_l[TDSIZE];
			_bncsve2_rtd_set0(&t0, &t1, &t2);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / _vl) * _vl;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += _vl)
			{
				svbool_t pg = svptrue_b64();
				svfloat64_t a_v = svld1_f64(pg, &(mat->element[_ti]));
				svint64_t   idx = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
				svfloat64_t b0  = svld1_gather_s64index_f64(pg, vec->element[0], idx);
				svfloat64_t b1  = svld1_gather_s64index_f64(pg, vec->element[1], idx);
				svfloat64_t b2  = svld1_gather_s64index_f64(pg, vec->element[2], idx);
				svfloat64_t zerov = svdup_n_f64(0.0);
				svfloat64_t m0, m1, m2;
				_bncsve2_rtd_mul(pg, &m0, &m1, &m2, a_v, zerov, zerov, b0, b1, b2);
				_bncsve2_rtd_add(pg, &t0, &t1, &t2, t0, t1, t2, m0, m1, m2);
				_ti += _vl;
			}
			mat_vec_i_l[0] = svaddv_f64(svptrue_b64(), t0);
			mat_vec_i_l[1] = svaddv_f64(svptrue_b64(), t1);
			mat_vec_i_l[2] = svaddv_f64(svptrue_b64(), t2);
			{
				double mij[TDSIZE], vj[TDSIZE], pr[TDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
					mij[0] = mat->element[_ti]; mij[1] = 0.0; mij[2] = 0.0;
					vj[0] = vec->element[0][idx_t];
					vj[1] = vec->element[1][idx_t];
					vj[2] = vec->element[2][idx_t];
					rtd_mul(pr, mij, vj);
					rtd_add(mat_vec_i_l, mat_vec_i_l, pr);
					_ti++;
				}
			}
			set_tdvector_i(ret, i, mat_vec_i_l);
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel for schedule(static) private(thread_index, j)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k, _ti;
			thread_index = omp_get_thread_num();
			_ti = 0;
			for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];

			float64x2_t a_td[TDSIZE], vb2[TDSIZE], tmp2[TDSIZE], tmp4_mul[TDSIZE];
			double mat_vec_i_l[TDSIZE], vset[2];
			_bncneon_rtd_set0(tmp2);
			a_td[1] = vdupq_n_f64(0.0); a_td[2] = vdupq_n_f64(0.0);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / 2) * 2;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += 2)
			{
				a_td[0] = vld1q_f64(&(mat->element[_ti]));
				vset[0] = vec->element[0][mat->nzero_index[i][j]]; vset[1] = vec->element[0][mat->nzero_index[i][j + 1]];
				vb2[0] = vld1q_f64(vset);
				vset[0] = vec->element[1][mat->nzero_index[i][j]]; vset[1] = vec->element[1][mat->nzero_index[i][j + 1]];
				vb2[1] = vld1q_f64(vset);
				vset[0] = vec->element[2][mat->nzero_index[i][j]]; vset[1] = vec->element[2][mat->nzero_index[i][j + 1]];
				vb2[2] = vld1q_f64(vset);
				_bncneon_rtd_mul(tmp4_mul, a_td, vb2);
				_bncneon_rtd_add(tmp2, tmp2, tmp4_mul);
				_ti += 2;
			}
			_bncneon_rtd_sum128d(mat_vec_i_l, tmp2);
			{
				double mij[TDSIZE], vj[TDSIZE], pr[TDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
					mij[0] = mat->element[_ti]; mij[1] = 0.0; mij[2] = 0.0;
					vj[0] = vec->element[0][idx_t];
					vj[1] = vec->element[1][idx_t];
					vj[2] = vec->element[2][idx_t];
					rtd_mul(pr, mij, vj);
					rtd_add(mat_vec_i_l, mat_vec_i_l, pr);
					_ti++;
				}
			}
			set_tdvector_i(ret, i, mat_vec_i_l);
		}
	}

#else // others

    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//get_mpfvector_i(ret, i) = 0.0;
		//mpf_set_ui(get_mpfvector_i(ret, i), 0UL);
		//rtd_set_ui(get_tdvector_i(ret, i), 0UL);
		rtd_set_ui(ret_i[thread_index], 0UL);

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpf_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			//rtd_mul(tmp, (mpf_ptr)(mat->element[total_index]), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mat_ij[thread_index] = mat->element[total_index[thread_index]];
			//mat_ij[thread_index][1] = mat->element[1][total_index[thread_index]];
			//mat_ij[thread_index][2] = mat->element[2][total_index[thread_index]];

			vec_j[thread_index][0] = vec->element[0][mat->nzero_index[i][j]];
			vec_j[thread_index][1] = vec->element[1][mat->nzero_index[i][j]];
			vec_j[thread_index][2] = vec->element[2][mat->nzero_index[i][j]];

			//printf("%d, %d: rtd_mul ", i, j);
			//rtd_mul(tmp[thread_index], mat_ij[thread_index], vec_j[thread_index]);
			rtd_mulq_d(tmp[thread_index], vec_j[thread_index], mat_ij[thread_index]);
			//mpf_add(get_mpfvector_i(ret, i), get_mpfvector_i(ret, i), tmp);
			//printf("rtd_add ");
			rtd_add(ret_i[thread_index], ret_i[thread_index], tmp[thread_index]);
			total_index[thread_index]++;
		}
		set_tdvector_i(ret, i, ret_i[thread_index]);
		//printf("set_tdvector_i %ld\n", i);
	}

#endif // __AVX2__

	return SUCCESS;
}

/* Multiply TDRSMatrix^T * TDVector */
int _bncomp_mul_tdrsmatrixt_tdvec(TDVector ret, TDRSMatrix mat, TDVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	//ddfloat tmp, mat_ji, vec_j, ret_j;
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], mat_ji[BNCOMP_MAX_NUM_THREADS][TDSIZE], vec_i[BNCOMP_MAX_NUM_THREADS][TDSIZE], ret_j[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	TDVector work_vec[BNCOMP_MAX_NUM_THREADS];
    int thread_index, num_threads;
	long int div_row, start_i, end_i;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//for(i = 0; i < mat->row_dim; i++)
	//	mpf_set_ui(get_mpfvector_i(ret, i), 0UL);

	// ret := 0
	set0_tdvector(ret);
	//total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS][TDSIZE], vb4[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][TDSIZE], ret4[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double vset[BNCOMP_MAX_NUM_THREADS][_BNC_D_WIDTH];

	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_tdvector(vec->dim);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		//for(i = 0; i < mat->row_dim; i++)
		for(i = start_i; i < end_i; i++)
		{
			thread_index = omp_get_thread_num();

			vb4[thread_index][0] = _mm256_set1_pd(vec->element[0][i]);
			vb4[thread_index][1] = _mm256_set1_pd(vec->element[1][i]);
			vb4[thread_index][2] = _mm256_set1_pd(vec->element[2][i]);

			//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
			jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//jmax = (mat->row_dim / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->row_dim % _BNC_D_WIDTH;
			//#pragma omp ordered
			for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
			{
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				a4[thread_index][0] = _mm256_load_pd(&(mat->element[0][total_index[thread_index]]));
				a4[thread_index][1] = _mm256_load_pd(&(mat->element[1][total_index[thread_index]]));
				a4[thread_index][2] = _mm256_load_pd(&(mat->element[2][total_index[thread_index]]));

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]];
				ret4[thread_index][0] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[0][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[0][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[0][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[0][mat->nzero_index[i][j    ]]
				);

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]];
				ret4[thread_index][1] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[1][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[1][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[1][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[1][mat->nzero_index[i][j    ]]
				);

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]];
				ret4[thread_index][2] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[1][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[1][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[1][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[1][mat->nzero_index[i][j    ]]
				);

				//ret4 = _mm256_fmadd_pd(a4, vb4, ret4);
				_bncavx2_rtd_mul(tmp4[thread_index], a4[thread_index], vb4[thread_index]);

			// #pragma omp critical
				_bncavx2_rtd_add(ret4[thread_index], ret4[thread_index], tmp4[thread_index]);

				if((j    ) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]] = ret4[thread_index][0][0];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]] = ret4[thread_index][1][0];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]] = ret4[thread_index][2][0];
				}
				if((j + 1) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]] = ret4[thread_index][0][1];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]] = ret4[thread_index][1][1];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]] = ret4[thread_index][2][1];
				}
				if((j + 2) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]] = ret4[thread_index][0][2];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]] = ret4[thread_index][1][2];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]] = ret4[thread_index][2][2];
				}
				if((j + 3) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]] = ret4[thread_index][0][3];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]] = ret4[thread_index][1][3];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]] = ret4[thread_index][2][3];
				}

				// total_index++;
				total_index[thread_index] += _BNC_D_WIDTH;
			}
			//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);

			//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		}
	} // #pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS][TDSIZE], vb8[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][TDSIZE], ret8[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_tdvector(vec->dim);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		//#pragma omp parallel for private(thread_index, j) ordered
		//for(i = 0; i < mat->row_dim; i++)
		for(i = start_i; i < end_i; i++)
		{
			thread_index = omp_get_thread_num();

			vb8[thread_index][0] = _mm512_set1_pd(vec->element[0][i]);
			vb8[thread_index][1] = _mm512_set1_pd(vec->element[1][i]);
			vb8[thread_index][2] = _mm512_set1_pd(vec->element[2][i]);

			//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
			jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//#pragma omp ordered
			for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
			{
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				a8[thread_index][0] = _mm512_load_pd(&(mat->element[0][total_index[thread_index]]));
				a8[thread_index][1] = _mm512_load_pd(&(mat->element[1][total_index[thread_index]]));
				a8[thread_index][2] = _mm512_load_pd(&(mat->element[2][total_index[thread_index]]));

				ret8[thread_index][0] = _mm512_set_pd(
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]]
				);
				ret8[thread_index][1] = _mm512_set_pd(
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]]
				);
				ret8[thread_index][2] = _mm512_set_pd(
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]]
				);

				//ret8 = _mm512_fmadd_pd(a8, vb8, ret8);
				_bncavx512_rtd_mul(tmp8[thread_index], a8[thread_index], vb8[thread_index]);

				//#pragma omp critical
				_bncavx512_rtd_add(ret8[thread_index], ret8[thread_index], tmp8[thread_index]);

				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 7]] = ret8[thread_index][0][7];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 6]] = ret8[thread_index][0][6];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 5]] = ret8[thread_index][0][5];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 4]] = ret8[thread_index][0][4];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]] = ret8[thread_index][0][3];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]] = ret8[thread_index][0][2];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]] = ret8[thread_index][0][1];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]] = ret8[thread_index][0][0];

				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 7]] = ret8[thread_index][1][7];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 6]] = ret8[thread_index][1][6];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 5]] = ret8[thread_index][1][5];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 4]] = ret8[thread_index][1][4];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]] = ret8[thread_index][1][3];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]] = ret8[thread_index][1][2];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]] = ret8[thread_index][1][1];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]] = ret8[thread_index][1][0];

				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 7]] = ret8[thread_index][2][7];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 6]] = ret8[thread_index][2][6];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 5]] = ret8[thread_index][2][5];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 4]] = ret8[thread_index][2][4];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]] = ret8[thread_index][2][3];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]] = ret8[thread_index][2][2];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]] = ret8[thread_index][2][1];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]] = ret8[thread_index][2][0];

				// total_index++;
				total_index[thread_index] += _BNC_D_WIDTH;
			}
			//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);

			//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP TD A^T*x: per-thread work_vec, gather/FMA/scatter (race-free). */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			work_vec[thread_index] = init_tdvector(vec->dim);
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;

			for(i = start_i; i < end_i; i++)
			{
				svfloat64_t vb0 = svdup_n_f64(vec->element[0][i]);
				svfloat64_t vb1 = svdup_n_f64(vec->element[1][i]);
				svfloat64_t vb2 = svdup_n_f64(vec->element[2][i]);
				long int _ti = 0, _k;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / _vl) * _vl;
				long int jres = nz - jmax;
				double *w0 = work_vec[thread_index]->element[0];
				double *w1 = work_vec[thread_index]->element[1];
				double *w2 = work_vec[thread_index]->element[2];
				for(j = 0; j < jmax; j += _vl)
				{
					svbool_t pg = svptrue_b64();
					svfloat64_t a0 = svld1_f64(pg, &(mat->element[0][_ti]));
					svfloat64_t a1 = svld1_f64(pg, &(mat->element[1][_ti]));
					svfloat64_t a2v= svld1_f64(pg, &(mat->element[2][_ti]));
					svint64_t idx  = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
					svfloat64_t r0 = svld1_gather_s64index_f64(pg, w0, idx);
					svfloat64_t r1 = svld1_gather_s64index_f64(pg, w1, idx);
					svfloat64_t r2 = svld1_gather_s64index_f64(pg, w2, idx);
					svfloat64_t m0, m1, m2;
					_bncsve2_rtd_mul(pg, &m0, &m1, &m2, a0, a1, a2v, vb0, vb1, vb2);
					_bncsve2_rtd_add(pg, &r0, &r1, &r2, r0, r1, r2, m0, m1, m2);
					svst1_scatter_s64index_f64(pg, w0, idx, r0);
					svst1_scatter_s64index_f64(pg, w1, idx, r1);
					svst1_scatter_s64index_f64(pg, w2, idx, r2);
					_ti += _vl;
				}
				{
					double mij[TDSIZE], vi[TDSIZE], pr[TDSIZE], cur[TDSIZE];
					long int col_dim = mat->nzero_col_dim[i];
					long int _t;
					vi[0]=vec->element[0][i]; vi[1]=vec->element[1][i]; vi[2]=vec->element[2][i];
					for(_t = 0; _t < jres; _t++)
					{
						long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
						mij[0]=mat->element[0][_ti]; mij[1]=mat->element[1][_ti]; mij[2]=mat->element[2][_ti];
						cur[0]=w0[idx_t]; cur[1]=w1[idx_t]; cur[2]=w2[idx_t];
						rtd_mul(pr, mij, vi);
						rtd_add(cur, cur, pr);
						w0[idx_t]=cur[0]; w1[idx_t]=cur[1]; w2[idx_t]=cur[2];
						_ti++;
					}
				}
			}
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			work_vec[thread_index] = init_tdvector(vec->dim);
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;

			for(i = start_i; i < end_i; i++)
			{
				float64x2_t vb2[TDSIZE], a2[TDSIZE], r2[TDSIZE], m2[TDSIZE];
				vb2[0]=vdupq_n_f64(vec->element[0][i]); vb2[1]=vdupq_n_f64(vec->element[1][i]); vb2[2]=vdupq_n_f64(vec->element[2][i]);
				long int _ti = 0, _k;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / 2) * 2;
				long int jres = nz - jmax;
				double *w0 = work_vec[thread_index]->element[0];
				double *w1 = work_vec[thread_index]->element[1];
				double *w2 = work_vec[thread_index]->element[2];
				double vset[2];
				for(j = 0; j < jmax; j += 2)
				{
					a2[0]=vld1q_f64(&(mat->element[0][_ti])); a2[1]=vld1q_f64(&(mat->element[1][_ti])); a2[2]=vld1q_f64(&(mat->element[2][_ti]));
					vset[0]=w0[mat->nzero_index[i][j]]; vset[1]=w0[mat->nzero_index[i][j + 1]]; r2[0]=vld1q_f64(vset);
					vset[0]=w1[mat->nzero_index[i][j]]; vset[1]=w1[mat->nzero_index[i][j + 1]]; r2[1]=vld1q_f64(vset);
					vset[0]=w2[mat->nzero_index[i][j]]; vset[1]=w2[mat->nzero_index[i][j + 1]]; r2[2]=vld1q_f64(vset);
					_bncneon_rtd_mul(m2, a2, vb2);
					_bncneon_rtd_add(r2, r2, m2);
					w0[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[0],0); w1[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[1],0); w2[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[2],0);
					w0[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[0],1); w1[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[1],1); w2[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[2],1);
					_ti += 2;
				}
				{
					double mij[TDSIZE], vi[TDSIZE], pr[TDSIZE], cur[TDSIZE];
					long int col_dim = mat->nzero_col_dim[i];
					long int _t;
					vi[0]=vec->element[0][i]; vi[1]=vec->element[1][i]; vi[2]=vec->element[2][i];
					for(_t = 0; _t < jres; _t++)
					{
						long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
						mij[0]=mat->element[0][_ti]; mij[1]=mat->element[1][_ti]; mij[2]=mat->element[2][_ti];
						cur[0]=w0[idx_t]; cur[1]=w1[idx_t]; cur[2]=w2[idx_t];
						rtd_mul(pr, mij, vi);
						rtd_add(cur, cur, pr);
						w0[idx_t]=cur[0]; w1[idx_t]=cur[1]; w2[idx_t]=cur[2];
						_ti++;
					}
				}
			}
		}
	}

#else // __AVX2__
	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_tdvector(vec->dim);

        //total_index[thread_index] = 0;

		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		for(i = start_i; i < end_i; i++)
		{
			//tmp_ret[0][i] = (double *)calloc(real_nzero_col_dim[i]);
			//tmp_ret[1][i] = (double *)calloc(real_nzero_col_dim[i]);

			thread_index = omp_get_thread_num();

			vec_i[thread_index][0] = vec->element[0][i];
			vec_i[thread_index][1] = vec->element[1][i];
			vec_i[thread_index][2] = vec->element[2][i];

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//#pragma omp ordered
			//ret_j[thread_index][0] = ret->element[0][i];
			//ret_j[thread_index][1] = ret->element[1][i];
			//rdd_set(ret_j, get_ddvector_i(ret, i));
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{

				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				mat_ji[thread_index][0] = mat->element[0][total_index[thread_index]];
				mat_ji[thread_index][1] = mat->element[1][total_index[thread_index]];
				mat_ji[thread_index][2] = mat->element[2][total_index[thread_index]];
				//ret_j[thread_index][0] = ret->element[0][mat->nzero_index[i][j]];
				//ret_j[thread_index][1] = ret->element[1][mat->nzero_index[i][j]];
				ret_j[thread_index][0] = work_vec[thread_index]->element[0][mat->nzero_index[i][j]];
				ret_j[thread_index][1] = work_vec[thread_index]->element[1][mat->nzero_index[i][j]];
				ret_j[thread_index][2] = work_vec[thread_index]->element[2][mat->nzero_index[i][j]];
				//rdd_mul(tmp, mat->element[total_index]), get_ddvector_i(vec, i));
				rtd_mul(tmp[thread_index], mat_ji[thread_index], vec_i[thread_index]);
				//rdd_add(ret->element[mat->nzero_index[i][j]], ret->element[mat->nzero_index[i][j]], tmp);

				//#pragma omp critical
				//{
				rtd_add(ret_j[thread_index], ret_j[thread_index], tmp[thread_index]);
				//set_ddvector_i(word, mat->nzero_index[i][j], ret_j[thread_index]);
				set_tdvector_i(work_vec[thread_index], mat->nzero_index[i][j], ret_j[thread_index]);
				//}
				total_index[thread_index]++;
			}
			//set_ddvector_i(ret, i, ret_j[thread_index]);
		}
	} // #omp parallel private(thread_index, num_threads, i, j)

#endif // __AVX2__

    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
		#pragma omp critical
		{
			add_tdvector(ret, ret, work_vec[thread_index]);
			free_tdvector(work_vec[thread_index]);
		}
	}


	return SUCCESS;
}

/* Multiply DRSMatrix^T * TDVector */
int _bncomp_mul_drsmatrixt_tdvec(TDVector ret, DRSMatrix mat, TDVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	//ddfloat tmp, mat_ji, vec_j, ret_j;
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], mat_ji[BNCOMP_MAX_NUM_THREADS], vec_i[BNCOMP_MAX_NUM_THREADS][TDSIZE], ret_j[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	TDVector work_vec[BNCOMP_MAX_NUM_THREADS];
    int thread_index, num_threads;
	long int div_row, start_i, end_i;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//for(i = 0; i < mat->row_dim; i++)
	//	mpf_set_ui(get_mpfvector_i(ret, i), 0UL);

	// ret := 0
	set0_tdvector(ret);
	//total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS], vb4[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][TDSIZE], ret4[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double vset[BNCOMP_MAX_NUM_THREADS][_BNC_D_WIDTH];

	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_tdvector(vec->dim);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		//for(i = 0; i < mat->row_dim; i++)
		for(i = start_i; i < end_i; i++)
		{
			thread_index = omp_get_thread_num();

			vb4[thread_index][0] = _mm256_set1_pd(vec->element[0][i]);
			vb4[thread_index][1] = _mm256_set1_pd(vec->element[1][i]);
			vb4[thread_index][2] = _mm256_set1_pd(vec->element[2][i]);

			//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
			jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//jmax = (mat->row_dim / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->row_dim % _BNC_D_WIDTH;
			//#pragma omp ordered
			for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
			{
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				a4[thread_index] = _mm256_load_pd(&(mat->element[total_index[thread_index]]));
				//a4[thread_index][1] = _mm256_load_pd(&(mat->element[1][total_index[thread_index]]));
				//a4[thread_index][2] = _mm256_load_pd(&(mat->element[2][total_index[thread_index]]));

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]];
				ret4[thread_index][0] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[0][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[0][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[0][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[0][mat->nzero_index[i][j    ]]
				);

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]];
				ret4[thread_index][1] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[1][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[1][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[1][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[1][mat->nzero_index[i][j    ]]
				);

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]];
				ret4[thread_index][2] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[1][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[1][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[1][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[1][mat->nzero_index[i][j    ]]
				);

				//ret4 = _mm256_fmadd_pd(a4, vb4, ret4);
				//_bncavx2_rtd_mul(tmp4[thread_index], a4[thread_index], vb4[thread_index]);
				_bncavx2_rtd_mulq_d(tmp4[thread_index], vb4[thread_index], a4[thread_index]);

			// #pragma omp critical
				_bncavx2_rtd_add(ret4[thread_index], ret4[thread_index], tmp4[thread_index]);

				if((j    ) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]] = ret4[thread_index][0][0];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]] = ret4[thread_index][1][0];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]] = ret4[thread_index][2][0];
				}
				if((j + 1) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]] = ret4[thread_index][0][1];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]] = ret4[thread_index][1][1];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]] = ret4[thread_index][2][1];
				}
				if((j + 2) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]] = ret4[thread_index][0][2];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]] = ret4[thread_index][1][2];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]] = ret4[thread_index][2][2];
				}
				if((j + 3) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]] = ret4[thread_index][0][3];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]] = ret4[thread_index][1][3];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]] = ret4[thread_index][2][3];
				}

				// total_index++;
				total_index[thread_index] += _BNC_D_WIDTH;
			}
			//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);

			//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		}
	} // #pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS], vb8[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][TDSIZE], ret8[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_tdvector(vec->dim);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		//#pragma omp parallel for private(thread_index, j) ordered
		//for(i = 0; i < mat->row_dim; i++)
		for(i = start_i; i < end_i; i++)
		{
			thread_index = omp_get_thread_num();

			vb8[thread_index][0] = _mm512_set1_pd(vec->element[0][i]);
			vb8[thread_index][1] = _mm512_set1_pd(vec->element[1][i]);
			vb8[thread_index][2] = _mm512_set1_pd(vec->element[2][i]);

			//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
			jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//#pragma omp ordered
			for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
			{
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				a8[thread_index] = _mm512_load_pd(&(mat->element[total_index[thread_index]]));
				//a8[thread_index][1] = _mm512_load_pd(&(mat->element[1][total_index[thread_index]]));
				//a8[thread_index][2] = _mm512_load_pd(&(mat->element[2][total_index[thread_index]]));

				ret8[thread_index][0] = _mm512_set_pd(
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]]
				);
				ret8[thread_index][1] = _mm512_set_pd(
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]]
				);
				ret8[thread_index][2] = _mm512_set_pd(
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]]
				);

				//ret8 = _mm512_fmadd_pd(a8, vb8, ret8);
				//_bncavx512_rtd_mul(tmp8[thread_index], a8[thread_index], vb8[thread_index]);
				_bncavx512_rtd_mul_d(tmp8[thread_index], vb8[thread_index], a8[thread_index]);				
				//_bncavx512_rtd_mul_d(tmp8[thread_index], a8[thread_index], vb8[thread_index]);

				//#pragma omp critical
				_bncavx512_rtd_add(ret8[thread_index], ret8[thread_index], tmp8[thread_index]);

				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 7]] = ret8[thread_index][0][7];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 6]] = ret8[thread_index][0][6];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 5]] = ret8[thread_index][0][5];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 4]] = ret8[thread_index][0][4];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]] = ret8[thread_index][0][3];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]] = ret8[thread_index][0][2];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]] = ret8[thread_index][0][1];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]] = ret8[thread_index][0][0];

				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 7]] = ret8[thread_index][1][7];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 6]] = ret8[thread_index][1][6];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 5]] = ret8[thread_index][1][5];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 4]] = ret8[thread_index][1][4];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]] = ret8[thread_index][1][3];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]] = ret8[thread_index][1][2];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]] = ret8[thread_index][1][1];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]] = ret8[thread_index][1][0];

				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 7]] = ret8[thread_index][2][7];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 6]] = ret8[thread_index][2][6];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 5]] = ret8[thread_index][2][5];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 4]] = ret8[thread_index][2][4];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]] = ret8[thread_index][2][3];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]] = ret8[thread_index][2][2];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]] = ret8[thread_index][2][1];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]] = ret8[thread_index][2][0];

				// total_index++;
				total_index[thread_index] += _BNC_D_WIDTH;
			}
			//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);

			//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP D^T*TD: mat plain double promoted to TD; per-thread work_vec. */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			work_vec[thread_index] = init_tdvector(vec->dim);
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;

			for(i = start_i; i < end_i; i++)
			{
				svfloat64_t vb0 = svdup_n_f64(vec->element[0][i]);
				svfloat64_t vb1 = svdup_n_f64(vec->element[1][i]);
				svfloat64_t vb2 = svdup_n_f64(vec->element[2][i]);
				svfloat64_t zerov = svdup_n_f64(0.0);
				long int _ti = 0, _k;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / _vl) * _vl;
				long int jres = nz - jmax;
				double *w0 = work_vec[thread_index]->element[0];
				double *w1 = work_vec[thread_index]->element[1];
				double *w2 = work_vec[thread_index]->element[2];
				for(j = 0; j < jmax; j += _vl)
				{
					svbool_t pg = svptrue_b64();
					svfloat64_t a_v = svld1_f64(pg, &(mat->element[_ti]));
					svint64_t idx  = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
					svfloat64_t r0 = svld1_gather_s64index_f64(pg, w0, idx);
					svfloat64_t r1 = svld1_gather_s64index_f64(pg, w1, idx);
					svfloat64_t r2 = svld1_gather_s64index_f64(pg, w2, idx);
					svfloat64_t m0, m1, m2;
					_bncsve2_rtd_mul(pg, &m0, &m1, &m2, a_v, zerov, zerov, vb0, vb1, vb2);
					_bncsve2_rtd_add(pg, &r0, &r1, &r2, r0, r1, r2, m0, m1, m2);
					svst1_scatter_s64index_f64(pg, w0, idx, r0);
					svst1_scatter_s64index_f64(pg, w1, idx, r1);
					svst1_scatter_s64index_f64(pg, w2, idx, r2);
					_ti += _vl;
				}
				{
					double mij[TDSIZE], vi[TDSIZE], pr[TDSIZE], cur[TDSIZE];
					long int col_dim = mat->nzero_col_dim[i];
					long int _t;
					vi[0]=vec->element[0][i]; vi[1]=vec->element[1][i]; vi[2]=vec->element[2][i];
					for(_t = 0; _t < jres; _t++)
					{
						long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
						mij[0]=mat->element[_ti]; mij[1]=0.0; mij[2]=0.0;
						cur[0]=w0[idx_t]; cur[1]=w1[idx_t]; cur[2]=w2[idx_t];
						rtd_mul(pr, mij, vi);
						rtd_add(cur, cur, pr);
						w0[idx_t]=cur[0]; w1[idx_t]=cur[1]; w2[idx_t]=cur[2];
						_ti++;
					}
				}
			}
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			work_vec[thread_index] = init_tdvector(vec->dim);
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;

			for(i = start_i; i < end_i; i++)
			{
				float64x2_t vb2[TDSIZE], r2[TDSIZE], m2[TDSIZE], a_td[TDSIZE];
				vb2[0]=vdupq_n_f64(vec->element[0][i]); vb2[1]=vdupq_n_f64(vec->element[1][i]); vb2[2]=vdupq_n_f64(vec->element[2][i]);
				a_td[1]=vdupq_n_f64(0.0); a_td[2]=vdupq_n_f64(0.0);
				long int _ti = 0, _k;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / 2) * 2;
				long int jres = nz - jmax;
				double *w0 = work_vec[thread_index]->element[0];
				double *w1 = work_vec[thread_index]->element[1];
				double *w2 = work_vec[thread_index]->element[2];
				double vset[2];
				for(j = 0; j < jmax; j += 2)
				{
					a_td[0] = vld1q_f64(&(mat->element[_ti]));
					vset[0]=w0[mat->nzero_index[i][j]]; vset[1]=w0[mat->nzero_index[i][j + 1]]; r2[0]=vld1q_f64(vset);
					vset[0]=w1[mat->nzero_index[i][j]]; vset[1]=w1[mat->nzero_index[i][j + 1]]; r2[1]=vld1q_f64(vset);
					vset[0]=w2[mat->nzero_index[i][j]]; vset[1]=w2[mat->nzero_index[i][j + 1]]; r2[2]=vld1q_f64(vset);
					_bncneon_rtd_mul(m2, a_td, vb2);
					_bncneon_rtd_add(r2, r2, m2);
					w0[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[0],0); w1[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[1],0); w2[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[2],0);
					w0[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[0],1); w1[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[1],1); w2[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[2],1);
					_ti += 2;
				}
				{
					double mij[TDSIZE], vi[TDSIZE], pr[TDSIZE], cur[TDSIZE];
					long int col_dim = mat->nzero_col_dim[i];
					long int _t;
					vi[0]=vec->element[0][i]; vi[1]=vec->element[1][i]; vi[2]=vec->element[2][i];
					for(_t = 0; _t < jres; _t++)
					{
						long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
						mij[0]=mat->element[_ti]; mij[1]=0.0; mij[2]=0.0;
						cur[0]=w0[idx_t]; cur[1]=w1[idx_t]; cur[2]=w2[idx_t];
						rtd_mul(pr, mij, vi);
						rtd_add(cur, cur, pr);
						w0[idx_t]=cur[0]; w1[idx_t]=cur[1]; w2[idx_t]=cur[2];
						_ti++;
					}
				}
			}
		}
	}

#else // __AVX2__
	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_tdvector(vec->dim);

        //total_index[thread_index] = 0;

		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		for(i = start_i; i < end_i; i++)
		{
			//tmp_ret[0][i] = (double *)calloc(real_nzero_col_dim[i]);
			//tmp_ret[1][i] = (double *)calloc(real_nzero_col_dim[i]);

			thread_index = omp_get_thread_num();

			vec_i[thread_index][0] = vec->element[0][i];
			vec_i[thread_index][1] = vec->element[1][i];
			vec_i[thread_index][2] = vec->element[2][i];

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//#pragma omp ordered
			//ret_j[thread_index][0] = ret->element[0][i];
			//ret_j[thread_index][1] = ret->element[1][i];
			//rdd_set(ret_j, get_ddvector_i(ret, i));
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{

				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				mat_ji[thread_index] = mat->element[total_index[thread_index]];
				//mat_ji[thread_index][1] = mat->element[1][total_index[thread_index]];
				//mat_ji[thread_index][2] = mat->element[2][total_index[thread_index]];
				//ret_j[thread_index][0] = ret->element[0][mat->nzero_index[i][j]];
				//ret_j[thread_index][1] = ret->element[1][mat->nzero_index[i][j]];
				ret_j[thread_index][0] = work_vec[thread_index]->element[0][mat->nzero_index[i][j]];
				ret_j[thread_index][1] = work_vec[thread_index]->element[1][mat->nzero_index[i][j]];
				ret_j[thread_index][2] = work_vec[thread_index]->element[2][mat->nzero_index[i][j]];
				//rdd_mul(tmp, mat->element[total_index]), get_ddvector_i(vec, i));
				//rtd_mul(tmp[thread_index], mat_ji[thread_index], vec_i[thread_index]);
				rtd_mulq_d(tmp[thread_index], vec_i[thread_index], mat_ji[thread_index]);
				//rdd_add(ret->element[mat->nzero_index[i][j]], ret->element[mat->nzero_index[i][j]], tmp);

				//#pragma omp critical
				//{
				rtd_add(ret_j[thread_index], ret_j[thread_index], tmp[thread_index]);
				//set_ddvector_i(word, mat->nzero_index[i][j], ret_j[thread_index]);
				set_tdvector_i(work_vec[thread_index], mat->nzero_index[i][j], ret_j[thread_index]);
				//}
				total_index[thread_index]++;
			}
			//set_ddvector_i(ret, i, ret_j[thread_index]);
		}
	} // #omp parallel private(thread_index, num_threads, i, j)

#endif // __AVX2__

    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
		#pragma omp critical
		{
			add_tdvector(ret, ret, work_vec[thread_index]);
			free_tdvector(work_vec[thread_index]);
		}
	}


	return SUCCESS;
}

/* Multiply CTDRSMatrix * CTDVector */
int _bncomp_mul_ctdrsmatrix_ctdvec(CTDVector ret, CTDRSMatrix mat, CTDVector vec)
{
	long int i, j, total_index;
	TDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_tdvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_tdrsmatrix_tdvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_tdrsmatrix_tdvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_tdrsmatrix_tdvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_tdrsmatrix_tdvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_sub_tdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_tdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_tdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CDRSMatrix * CTDVector */
int _bncomp_mul_cdrsmatrix_ctdvec(CTDVector ret, CDRSMatrix mat, CTDVector vec)
{
	long int i, j, total_index;
	TDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_tdvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_drsmatrix_tdvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_drsmatrix_tdvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_drsmatrix_tdvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_drsmatrix_tdvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_sub_tdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_tdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_tdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CTDRSMatrix^T * CTDVector */
int _bncomp_mul_ctdrsmatrixt_ctdvec(CTDVector ret, CTDRSMatrix mat, CTDVector vec)
{
	long int i, j, total_index;
	TDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_tdvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_tdrsmatrixt_tdvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_tdrsmatrixt_tdvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_tdrsmatrixt_tdvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_tdrsmatrixt_tdvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_sub_tdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_tdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_tdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CDRSMatrix^T * CTDVector */
int _bncomp_mul_cdrsmatrixt_ctdvec(CTDVector ret, CDRSMatrix mat, CTDVector vec)
{
	long int i, j, total_index;
	TDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_tdvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_drsmatrixt_tdvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_drsmatrixt_tdvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_drsmatrixt_tdvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_drsmatrixt_tdvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_sub_tdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_tdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_tdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply conj(CTDRSMatrix)^T * CTDVector */
int _bncomp_mul_ctdrsmatrixs_ctdvec(CTDVector ret, CTDRSMatrix mat, CTDVector vec)
{
	long int i, j, total_index;
	TDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_tdvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_tdrsmatrixt_tdvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_tdrsmatrixt_tdvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_tdrsmatrixt_tdvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_tdrsmatrixt_tdvec(tmp_vec[3], mat->im, vec->im);

	// mat_re * vec_re + mat_im * vec_im
	_bncomp_add_tdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	// mat_re * vec_im - mat_im * vec_re
	_bncomp_sub_tdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_tdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply conj(CDRSMatrix)^T * CTDVector */
int _bncomp_mul_cdrsmatrixs_ctdvec(CTDVector ret, CDRSMatrix mat, CTDVector vec)
{
	long int i, j, total_index;
	TDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_tdvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_drsmatrixt_tdvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_drsmatrixt_tdvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_drsmatrixt_tdvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_drsmatrixt_tdvec(tmp_vec[3], mat->im, vec->im);

	// mat_re * vec_re + mat_im * vec_im
	_bncomp_add_tdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	// mat_re * vec_im - mat_im * vec_re
	_bncomp_sub_tdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_tdvector(tmp_vec[i]);

	return SUCCESS;
}


//-----------------------------------------------
// QD precision
//-----------------------------------------------

/* Multiply QDRSMatrix * QDVector */
int _bncomp_mul_qdrsmatrix_qdvec(QDVector ret, QDRSMatrix mat, QDVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	//qdfloat tmp, mat_ij, vec_j, ret_i;
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE], mat_ij[BNCOMP_MAX_NUM_THREADS][QDSIZE], vec_j[BNCOMP_MAX_NUM_THREADS][QDSIZE], ret_i[BNCOMP_MAX_NUM_THREADS][QDSIZE];
    int thread_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS][QDSIZE], vb4[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp4_mul[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double mat_vec_i[BNCOMP_MAX_NUM_THREADS][QDSIZE], vset[BNCOMP_MAX_NUM_THREADS][_BNC_D_WIDTH];

    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//ret->element[i] = 0.0;
		//tmp4 = _mm256_setzero_pd();
		_bncavx2_set0_qd(tmp4[thread_index]);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		// jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		// jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a4[thread_index][0]  = _mm256_load_pd(&(mat->element[0][total_index[thread_index]]));
			a4[thread_index][1]  = _mm256_load_pd(&(mat->element[1][total_index[thread_index]]));
			a4[thread_index][2]  = _mm256_load_pd(&(mat->element[2][total_index[thread_index]]));
			a4[thread_index][3]  = _mm256_load_pd(&(mat->element[3][total_index[thread_index]]));

			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[0][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[0][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[0][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[0][mat->nzero_index[i][j + 3]];
			//printf("set0 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][0] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[0][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[0][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[0][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[0][mat->nzero_index[i][j    ]]
			);

			//printf("set1 ");
			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[1][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[1][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[1][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[1][mat->nzero_index[i][j + 3]];
			//("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][1] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);

			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[2][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[2][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[2][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[2][mat->nzero_index[i][j + 3]];
			//printf("set0 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][2] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[0][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[0][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[0][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[0][mat->nzero_index[i][j    ]]
			);

			//printf("set1 ");
			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[3][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[3][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[3][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[3][mat->nzero_index[i][j + 3]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][3] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);

			//tmp4 = _mm256_fmadd_pd(a4, vb4, tmp4);
			_bncavx2_rqd_mul(tmp4_mul[thread_index], a4[thread_index], vb4[thread_index]);
			_bncavx2_rqd_add(tmp4[thread_index], tmp4[thread_index], tmp4_mul[thread_index]);

			// total_index++;
			total_index[thread_index] += _BNC_D_WIDTH;
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];

		_bncavx2_rqd_sum256d(mat_vec_i[thread_index], tmp4[thread_index]);

		set_qdvector_i(ret, i, mat_vec_i[thread_index]);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS][QDSIZE], vb8[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp8_mul[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double mat_vec_i[BNCOMP_MAX_NUM_THREADS][DDSIZE];

    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//ret->element[i] = 0.0;
		//tmp8 = _mm512_setzero_pd();
		_bncavx512_set0_qd(tmp8[thread_index]);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a8[thread_index][0] = _mm512_load_pd(&(mat->element[0][total_index[thread_index]]));
			a8[thread_index][1] = _mm512_load_pd(&(mat->element[1][total_index[thread_index]]));
			a8[thread_index][2] = _mm512_load_pd(&(mat->element[2][total_index[thread_index]]));
			a8[thread_index][3] = _mm512_load_pd(&(mat->element[3][total_index[thread_index]]));

			vb8[thread_index][0] = _mm512_set_pd(
				vec->element[0][mat->nzero_index[i][j + 7]],
				vec->element[0][mat->nzero_index[i][j + 6]],
				vec->element[0][mat->nzero_index[i][j + 5]],
				vec->element[0][mat->nzero_index[i][j + 4]],
				vec->element[0][mat->nzero_index[i][j + 3]],
				vec->element[0][mat->nzero_index[i][j + 2]],
				vec->element[0][mat->nzero_index[i][j + 1]],
				vec->element[0][mat->nzero_index[i][j    ]]
			);
			vb8[thread_index][1] = _mm512_set_pd(
				vec->element[1][mat->nzero_index[i][j + 7]],
				vec->element[1][mat->nzero_index[i][j + 6]],
				vec->element[1][mat->nzero_index[i][j + 5]],
				vec->element[1][mat->nzero_index[i][j + 4]],
				vec->element[1][mat->nzero_index[i][j + 3]],
				vec->element[1][mat->nzero_index[i][j + 2]],
				vec->element[1][mat->nzero_index[i][j + 1]],
				vec->element[1][mat->nzero_index[i][j    ]]
			);
			vb8[thread_index][2] = _mm512_set_pd(
				vec->element[2][mat->nzero_index[i][j + 7]],
				vec->element[2][mat->nzero_index[i][j + 6]],
				vec->element[2][mat->nzero_index[i][j + 5]],
				vec->element[2][mat->nzero_index[i][j + 4]],
				vec->element[2][mat->nzero_index[i][j + 3]],
				vec->element[2][mat->nzero_index[i][j + 2]],
				vec->element[2][mat->nzero_index[i][j + 1]],
				vec->element[2][mat->nzero_index[i][j    ]]
			);
			vb8[thread_index][3] = _mm512_set_pd(
				vec->element[3][mat->nzero_index[i][j + 7]],
				vec->element[3][mat->nzero_index[i][j + 6]],
				vec->element[3][mat->nzero_index[i][j + 5]],
				vec->element[3][mat->nzero_index[i][j + 4]],
				vec->element[3][mat->nzero_index[i][j + 3]],
				vec->element[3][mat->nzero_index[i][j + 2]],
				vec->element[3][mat->nzero_index[i][j + 1]],
				vec->element[3][mat->nzero_index[i][j    ]]
			);

			//tmp8 = _mm512_fmadd_pd(a4, vb4, tmp4);
			_bncavx512_rqd_mul(tmp8_mul[thread_index], a8[thread_index], vb8[thread_index]);
			_bncavx512_rqd_add(tmp8[thread_index], tmp8[thread_index], tmp8_mul[thread_index]);

			// total_index++;
			total_index[thread_index] += _BNC_D_WIDTH;
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];

		_bncavx512_rqd_sum512d(mat_vec_i[thread_index], tmp8[thread_index]);

		set_qdvector_i(ret, i, mat_vec_i[thread_index]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP QD SpMV. */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel for schedule(static) private(thread_index, j)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k, _ti;
			thread_index = omp_get_thread_num();
			_ti = 0;
			for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];

			svfloat64_t t0, t1, t2, t3;
			double mat_vec_i_l[QDSIZE];
			_bncsve2_rqd_set0(&t0, &t1, &t2, &t3);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / _vl) * _vl;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += _vl)
			{
				svbool_t pg = svptrue_b64();
				svfloat64_t a0 = svld1_f64(pg, &(mat->element[0][_ti]));
				svfloat64_t a1 = svld1_f64(pg, &(mat->element[1][_ti]));
				svfloat64_t a2v= svld1_f64(pg, &(mat->element[2][_ti]));
				svfloat64_t a3 = svld1_f64(pg, &(mat->element[3][_ti]));
				svint64_t   idx= svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
				svfloat64_t b0 = svld1_gather_s64index_f64(pg, vec->element[0], idx);
				svfloat64_t b1 = svld1_gather_s64index_f64(pg, vec->element[1], idx);
				svfloat64_t b2 = svld1_gather_s64index_f64(pg, vec->element[2], idx);
				svfloat64_t b3 = svld1_gather_s64index_f64(pg, vec->element[3], idx);
				svfloat64_t m0, m1, m2, m3;
				_bncsve2_rqd_mul(pg, &m0, &m1, &m2, &m3, a0, a1, a2v, a3, b0, b1, b2, b3);
				_bncsve2_rqd_add(pg, &t0, &t1, &t2, &t3, t0, t1, t2, t3, m0, m1, m2, m3);
				_ti += _vl;
			}
			mat_vec_i_l[0] = svaddv_f64(svptrue_b64(), t0);
			mat_vec_i_l[1] = svaddv_f64(svptrue_b64(), t1);
			mat_vec_i_l[2] = svaddv_f64(svptrue_b64(), t2);
			mat_vec_i_l[3] = svaddv_f64(svptrue_b64(), t3);
			{
				double mij[QDSIZE], vj[QDSIZE], pr[QDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
					mij[0] = mat->element[0][_ti];
					mij[1] = mat->element[1][_ti];
					mij[2] = mat->element[2][_ti];
					mij[3] = mat->element[3][_ti];
					vj[0]  = vec->element[0][idx_t];
					vj[1]  = vec->element[1][idx_t];
					vj[2]  = vec->element[2][idx_t];
					vj[3]  = vec->element[3][idx_t];
					rqd_mul(pr, mij, vj);
					rqd_add(mat_vec_i_l, mat_vec_i_l, pr);
					_ti++;
				}
			}
			set_qdvector_i(ret, i, mat_vec_i_l);
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel for schedule(static) private(thread_index, j)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k, _ti;
			thread_index = omp_get_thread_num();
			_ti = 0;
			for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];

			float64x2_t a2[QDSIZE], vb2[QDSIZE], tmp2[QDSIZE], tmp4_mul[QDSIZE];
			double mat_vec_i_l[QDSIZE], vset[2];
			_bncneon_set0_qd(tmp2);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / 2) * 2;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += 2)
			{
				a2[0] = vld1q_f64(&(mat->element[0][_ti]));
				a2[1] = vld1q_f64(&(mat->element[1][_ti]));
				a2[2] = vld1q_f64(&(mat->element[2][_ti]));
				a2[3] = vld1q_f64(&(mat->element[3][_ti]));
				vset[0] = vec->element[0][mat->nzero_index[i][j    ]];
				vset[1] = vec->element[0][mat->nzero_index[i][j + 1]];
				vb2[0] = vld1q_f64(vset);
				vset[0] = vec->element[1][mat->nzero_index[i][j    ]];
				vset[1] = vec->element[1][mat->nzero_index[i][j + 1]];
				vb2[1] = vld1q_f64(vset);
				vset[0] = vec->element[2][mat->nzero_index[i][j    ]];
				vset[1] = vec->element[2][mat->nzero_index[i][j + 1]];
				vb2[2] = vld1q_f64(vset);
				vset[0] = vec->element[3][mat->nzero_index[i][j    ]];
				vset[1] = vec->element[3][mat->nzero_index[i][j + 1]];
				vb2[3] = vld1q_f64(vset);
				_bncneon_rqd_mul(tmp4_mul, a2, vb2);
				_bncneon_rqd_add(tmp2, tmp2, tmp4_mul);
				_ti += 2;
			}
			_bncneon_rqd_sum128d(mat_vec_i_l, tmp2);
			{
				double mij[QDSIZE], vj[QDSIZE], pr[QDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
					mij[0] = mat->element[0][_ti];
					mij[1] = mat->element[1][_ti];
					mij[2] = mat->element[2][_ti];
					mij[3] = mat->element[3][_ti];
					vj[0]  = vec->element[0][idx_t];
					vj[1]  = vec->element[1][idx_t];
					vj[2]  = vec->element[2][idx_t];
					vj[3]  = vec->element[3][idx_t];
					rqd_mul(pr, mij, vj);
					rqd_add(mat_vec_i_l, mat_vec_i_l, pr);
					_ti++;
				}
			}
			set_qdvector_i(ret, i, mat_vec_i_l);
		}
	}

#else // others
    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//get_mpfvector_i(ret, i) = 0.0;
		//mpf_set_ui(get_mpfvector_i(ret, i), 0UL);
		//rqd_set_ui(get_qdvector_i(ret, i), 0UL);
		rqd_set_ui(ret_i[thread_index], 0UL);

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpf_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			//rqd_mul(tmp, (mpf_ptr)(mat->element[total_index]), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mat_ij[thread_index][0] = mat->element[0][total_index[thread_index]];
			mat_ij[thread_index][1] = mat->element[1][total_index[thread_index]];
			mat_ij[thread_index][2] = mat->element[2][total_index[thread_index]];
			mat_ij[thread_index][3] = mat->element[3][total_index[thread_index]];

			vec_j[thread_index][0] = vec->element[0][mat->nzero_index[i][j]];
			vec_j[thread_index][1] = vec->element[1][mat->nzero_index[i][j]];
			vec_j[thread_index][2] = vec->element[2][mat->nzero_index[i][j]];
			vec_j[thread_index][3] = vec->element[3][mat->nzero_index[i][j]];

			rqd_mul(tmp[thread_index], mat_ij[thread_index], vec_j[thread_index]);
			//mpf_add(get_mpfvector_i(ret, i), get_mpfvector_i(ret, i), tmp);
			rqd_add(ret_i[thread_index], ret_i[thread_index], tmp[thread_index]);
			total_index[thread_index]++;
		}
		set_qdvector_i(ret, i, ret_i[thread_index]);
	}

#endif // __AVX2__

	return SUCCESS;
}

/* Multiply DRSMatrix * QDVector */
int _bncomp_mul_drsmatrix_qdvec(QDVector ret, DRSMatrix mat, QDVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	//qdfloat tmp, mat_ij, vec_j, ret_i;
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE], mat_ij[BNCOMP_MAX_NUM_THREADS], vec_j[BNCOMP_MAX_NUM_THREADS][QDSIZE], ret_i[BNCOMP_MAX_NUM_THREADS][QDSIZE];
    int thread_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS], vb4[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp4_mul[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double mat_vec_i[BNCOMP_MAX_NUM_THREADS][QDSIZE], vset[BNCOMP_MAX_NUM_THREADS][_BNC_D_WIDTH];

    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//ret->element[i] = 0.0;
		//tmp4 = _mm256_setzero_pd();
		_bncavx2_set0_qd(tmp4[thread_index]);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		// jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		// jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a4[thread_index]  = _mm256_load_pd(&(mat->element[total_index[thread_index]]));
			//a4[thread_index][1]  = _mm256_load_pd(&(mat->element[1][total_index[thread_index]]));
			//a4[thread_index][2]  = _mm256_load_pd(&(mat->element[2][total_index[thread_index]]));
			//a4[thread_index][3]  = _mm256_load_pd(&(mat->element[3][total_index[thread_index]]));

			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[0][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[0][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[0][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[0][mat->nzero_index[i][j + 3]];
			//printf("set0 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][0] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[0][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[0][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[0][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[0][mat->nzero_index[i][j    ]]
			);

			//printf("set1 ");
			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[1][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[1][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[1][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[1][mat->nzero_index[i][j + 3]];
			//("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][1] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);

			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[2][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[2][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[2][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[2][mat->nzero_index[i][j + 3]];
			//printf("set0 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][2] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[0][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[0][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[0][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[0][mat->nzero_index[i][j    ]]
			);

			//printf("set1 ");
			vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = vec->element[3][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = vec->element[3][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = vec->element[3][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = vec->element[3][mat->nzero_index[i][j + 3]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[thread_index][3] = _mm256_set_pd(
				vset[thread_index][3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[thread_index][2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[thread_index][1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[thread_index][0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);

			//tmp4 = _mm256_fmadd_pd(a4, vb4, tmp4);
			//_bncavx2_rqd_mul(tmp4_mul[thread_index], a4[thread_index], vb4[thread_index]);
			_bncavx2_rqd_mul_d(tmp4_mul[thread_index], vb4[thread_index], a4[thread_index]);
			_bncavx2_rqd_add(tmp4[thread_index], tmp4[thread_index], tmp4_mul[thread_index]);

			// total_index++;
			total_index[thread_index] += _BNC_D_WIDTH;
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];

		_bncavx2_rqd_sum256d(mat_vec_i[thread_index], tmp4[thread_index]);

		set_qdvector_i(ret, i, mat_vec_i[thread_index]);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS], vb8[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp8_mul[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double mat_vec_i[BNCOMP_MAX_NUM_THREADS][DDSIZE];

    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//ret->element[i] = 0.0;
		//tmp8 = _mm512_setzero_pd();
		_bncavx512_set0_qd(tmp8[thread_index]);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a8[thread_index] = _mm512_load_pd(&(mat->element[total_index[thread_index]]));
			//a8[thread_index][1] = _mm512_load_pd(&(mat->element[1][total_index[thread_index]]));
			//a8[thread_index][2] = _mm512_load_pd(&(mat->element[2][total_index[thread_index]]));
			//a8[thread_index][3] = _mm512_load_pd(&(mat->element[3][total_index[thread_index]]));

			vb8[thread_index][0] = _mm512_set_pd(
				vec->element[0][mat->nzero_index[i][j + 7]],
				vec->element[0][mat->nzero_index[i][j + 6]],
				vec->element[0][mat->nzero_index[i][j + 5]],
				vec->element[0][mat->nzero_index[i][j + 4]],
				vec->element[0][mat->nzero_index[i][j + 3]],
				vec->element[0][mat->nzero_index[i][j + 2]],
				vec->element[0][mat->nzero_index[i][j + 1]],
				vec->element[0][mat->nzero_index[i][j    ]]
			);
			vb8[thread_index][1] = _mm512_set_pd(
				vec->element[1][mat->nzero_index[i][j + 7]],
				vec->element[1][mat->nzero_index[i][j + 6]],
				vec->element[1][mat->nzero_index[i][j + 5]],
				vec->element[1][mat->nzero_index[i][j + 4]],
				vec->element[1][mat->nzero_index[i][j + 3]],
				vec->element[1][mat->nzero_index[i][j + 2]],
				vec->element[1][mat->nzero_index[i][j + 1]],
				vec->element[1][mat->nzero_index[i][j    ]]
			);
			vb8[thread_index][2] = _mm512_set_pd(
				vec->element[2][mat->nzero_index[i][j + 7]],
				vec->element[2][mat->nzero_index[i][j + 6]],
				vec->element[2][mat->nzero_index[i][j + 5]],
				vec->element[2][mat->nzero_index[i][j + 4]],
				vec->element[2][mat->nzero_index[i][j + 3]],
				vec->element[2][mat->nzero_index[i][j + 2]],
				vec->element[2][mat->nzero_index[i][j + 1]],
				vec->element[2][mat->nzero_index[i][j    ]]
			);
			vb8[thread_index][3] = _mm512_set_pd(
				vec->element[3][mat->nzero_index[i][j + 7]],
				vec->element[3][mat->nzero_index[i][j + 6]],
				vec->element[3][mat->nzero_index[i][j + 5]],
				vec->element[3][mat->nzero_index[i][j + 4]],
				vec->element[3][mat->nzero_index[i][j + 3]],
				vec->element[3][mat->nzero_index[i][j + 2]],
				vec->element[3][mat->nzero_index[i][j + 1]],
				vec->element[3][mat->nzero_index[i][j    ]]
			);

			//tmp8 = _mm512_fmadd_pd(a4, vb4, tmp4);
			//_bncavx512_rqd_mul(tmp8_mul[thread_index], a8[thread_index], vb8[thread_index]);
			_bncavx512_rqd_mul_d(tmp8_mul[thread_index], vb8[thread_index], a8[thread_index]);
			_bncavx512_rqd_add(tmp8[thread_index], tmp8[thread_index], tmp8_mul[thread_index]);

			// total_index++;
			total_index[thread_index] += _BNC_D_WIDTH;
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];

		_bncavx512_rqd_sum512d(mat_vec_i[thread_index], tmp8[thread_index]);

		set_qdvector_i(ret, i, mat_vec_i[thread_index]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP D*QD SpMV: promote plain-double mat to QD (a,0,0,0). */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel for schedule(static) private(thread_index, j)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k, _ti;
			thread_index = omp_get_thread_num();
			_ti = 0;
			for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];

			svfloat64_t t0, t1, t2, t3;
			double mat_vec_i_l[QDSIZE];
			_bncsve2_rqd_set0(&t0, &t1, &t2, &t3);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / _vl) * _vl;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += _vl)
			{
				svbool_t pg = svptrue_b64();
				svfloat64_t a_v = svld1_f64(pg, &(mat->element[_ti]));
				svint64_t   idx = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
				svfloat64_t b0  = svld1_gather_s64index_f64(pg, vec->element[0], idx);
				svfloat64_t b1  = svld1_gather_s64index_f64(pg, vec->element[1], idx);
				svfloat64_t b2  = svld1_gather_s64index_f64(pg, vec->element[2], idx);
				svfloat64_t b3  = svld1_gather_s64index_f64(pg, vec->element[3], idx);
				svfloat64_t zerov = svdup_n_f64(0.0);
				svfloat64_t m0, m1, m2, m3;
				_bncsve2_rqd_mul(pg, &m0, &m1, &m2, &m3, a_v, zerov, zerov, zerov, b0, b1, b2, b3);
				_bncsve2_rqd_add(pg, &t0, &t1, &t2, &t3, t0, t1, t2, t3, m0, m1, m2, m3);
				_ti += _vl;
			}
			mat_vec_i_l[0] = svaddv_f64(svptrue_b64(), t0);
			mat_vec_i_l[1] = svaddv_f64(svptrue_b64(), t1);
			mat_vec_i_l[2] = svaddv_f64(svptrue_b64(), t2);
			mat_vec_i_l[3] = svaddv_f64(svptrue_b64(), t3);
			{
				double mij[QDSIZE], vj[QDSIZE], pr[QDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
					mij[0]=mat->element[_ti]; mij[1]=0.0; mij[2]=0.0; mij[3]=0.0;
					vj[0]=vec->element[0][idx_t]; vj[1]=vec->element[1][idx_t]; vj[2]=vec->element[2][idx_t]; vj[3]=vec->element[3][idx_t];
					rqd_mul(pr, mij, vj);
					rqd_add(mat_vec_i_l, mat_vec_i_l, pr);
					_ti++;
				}
			}
			set_qdvector_i(ret, i, mat_vec_i_l);
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel for schedule(static) private(thread_index, j)
		for(i = 0; i < mat->row_dim; i++)
		{
			long int _k, _ti;
			thread_index = omp_get_thread_num();
			_ti = 0;
			for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];

			float64x2_t a_qd[QDSIZE], vb2[QDSIZE], tmp2[QDSIZE], tmp4_mul[QDSIZE];
			double mat_vec_i_l[QDSIZE], vset[2];
			_bncneon_rqd_set0(tmp2);
			a_qd[1]=vdupq_n_f64(0.0); a_qd[2]=vdupq_n_f64(0.0); a_qd[3]=vdupq_n_f64(0.0);
			long int nz = mat->nzero_col_dim[i];
			long int jmax = (nz / 2) * 2;
			long int jres = nz - jmax;
			for(j = 0; j < jmax; j += 2)
			{
				a_qd[0] = vld1q_f64(&(mat->element[_ti]));
				vset[0]=vec->element[0][mat->nzero_index[i][j]]; vset[1]=vec->element[0][mat->nzero_index[i][j + 1]]; vb2[0]=vld1q_f64(vset);
				vset[0]=vec->element[1][mat->nzero_index[i][j]]; vset[1]=vec->element[1][mat->nzero_index[i][j + 1]]; vb2[1]=vld1q_f64(vset);
				vset[0]=vec->element[2][mat->nzero_index[i][j]]; vset[1]=vec->element[2][mat->nzero_index[i][j + 1]]; vb2[2]=vld1q_f64(vset);
				vset[0]=vec->element[3][mat->nzero_index[i][j]]; vset[1]=vec->element[3][mat->nzero_index[i][j + 1]]; vb2[3]=vld1q_f64(vset);
				_bncneon_rqd_mul(tmp4_mul, a_qd, vb2);
				_bncneon_rqd_add(tmp2, tmp2, tmp4_mul);
				_ti += 2;
			}
			_bncneon_rqd_sum128d(mat_vec_i_l, tmp2);
			{
				double mij[QDSIZE], vj[QDSIZE], pr[QDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				long int _t;
				for(_t = 0; _t < jres; _t++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
					mij[0]=mat->element[_ti]; mij[1]=0.0; mij[2]=0.0; mij[3]=0.0;
					vj[0]=vec->element[0][idx_t]; vj[1]=vec->element[1][idx_t]; vj[2]=vec->element[2][idx_t]; vj[3]=vec->element[3][idx_t];
					rqd_mul(pr, mij, vj);
					rqd_add(mat_vec_i_l, mat_vec_i_l, pr);
					_ti++;
				}
			}
			set_qdvector_i(ret, i, mat_vec_i_l);
		}
	}

#else // others
    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

		//get_mpfvector_i(ret, i) = 0.0;
		//mpf_set_ui(get_mpfvector_i(ret, i), 0UL);
		//rqd_set_ui(get_qdvector_i(ret, i), 0UL);
		rqd_set_ui(ret_i[thread_index], 0UL);

        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpf_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			//rqd_mul(tmp, (mpf_ptr)(mat->element[total_index]), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mat_ij[thread_index] = mat->element[total_index[thread_index]];
			//mat_ij[thread_index][1] = mat->element[1][total_index[thread_index]];
			//mat_ij[thread_index][2] = mat->element[2][total_index[thread_index]];
			//mat_ij[thread_index][3] = mat->element[3][total_index[thread_index]];

			vec_j[thread_index][0] = vec->element[0][mat->nzero_index[i][j]];
			vec_j[thread_index][1] = vec->element[1][mat->nzero_index[i][j]];
			vec_j[thread_index][2] = vec->element[2][mat->nzero_index[i][j]];
			vec_j[thread_index][3] = vec->element[3][mat->nzero_index[i][j]];

			//rqd_mul(tmp[thread_index], mat_ij[thread_index], vec_j[thread_index]);
			rqd_mul_d(tmp[thread_index], vec_j[thread_index], mat_ij[thread_index]);
			//mpf_add(get_mpfvector_i(ret, i), get_mpfvector_i(ret, i), tmp);
			rqd_add(ret_i[thread_index], ret_i[thread_index], tmp[thread_index]);
			total_index[thread_index]++;
		}
		set_qdvector_i(ret, i, ret_i[thread_index]);
	}

#endif // __AVX2__

	return SUCCESS;
}

/* Multiply QDRSMatrix^T * QDVector */
int _bncomp_mul_qdrsmatrixt_qdvec(QDVector ret, QDRSMatrix mat, QDVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	//ddfloat tmp, mat_ji, vec_j, ret_j;
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE], mat_ji[BNCOMP_MAX_NUM_THREADS][QDSIZE], vec_i[BNCOMP_MAX_NUM_THREADS][QDSIZE], ret_j[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	QDVector work_vec[BNCOMP_MAX_NUM_THREADS];
    int thread_index, num_threads;
	long int div_row, start_i, end_i;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//for(i = 0; i < mat->row_dim; i++)
	//	mpf_set_ui(get_mpfvector_i(ret, i), 0UL);

	// ret := 0
	set0_qdvector(ret);
	//total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS][QDSIZE], vb4[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][QDSIZE], ret4[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double vset[BNCOMP_MAX_NUM_THREADS][_BNC_D_WIDTH];

	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_qdvector(vec->dim);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		//for(i = 0; i < mat->row_dim; i++)
		for(i = start_i; i < end_i; i++)
		{
			thread_index = omp_get_thread_num();

			vb4[thread_index][0] = _mm256_set1_pd(vec->element[0][i]);
			vb4[thread_index][1] = _mm256_set1_pd(vec->element[1][i]);
			vb4[thread_index][2] = _mm256_set1_pd(vec->element[2][i]);
			vb4[thread_index][3] = _mm256_set1_pd(vec->element[3][i]);

			//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
			jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//jmax = (mat->row_dim / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->row_dim % _BNC_D_WIDTH;
			//#pragma omp ordered
			for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
			{
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				a4[thread_index][0] = _mm256_load_pd(&(mat->element[0][total_index[thread_index]]));
				a4[thread_index][1] = _mm256_load_pd(&(mat->element[1][total_index[thread_index]]));
				a4[thread_index][2] = _mm256_load_pd(&(mat->element[2][total_index[thread_index]]));
				a4[thread_index][3] = _mm256_load_pd(&(mat->element[3][total_index[thread_index]]));

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]];
				ret4[thread_index][0] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[0][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[0][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[0][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[0][mat->nzero_index[i][j    ]]
				);

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]];
				ret4[thread_index][1] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[1][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[1][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[1][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[1][mat->nzero_index[i][j    ]]
				);

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]];
				ret4[thread_index][2] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[1][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[1][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[1][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[1][mat->nzero_index[i][j    ]]
				);

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[3][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[3][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[3][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[3][mat->nzero_index[i][j + 3]];
				ret4[thread_index][3] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[1][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[1][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[1][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[1][mat->nzero_index[i][j    ]]
				);
				//ret4 = _mm256_fmadd_pd(a4, vb4, ret4);
				_bncavx2_rqd_mul(tmp4[thread_index], a4[thread_index], vb4[thread_index]);

			// #pragma omp critical
				_bncavx2_rqd_add(ret4[thread_index], ret4[thread_index], tmp4[thread_index]);

				if((j    ) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]] = ret4[thread_index][0][0];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]] = ret4[thread_index][1][0];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]] = ret4[thread_index][2][0];
					work_vec[thread_index]->element[3][mat->nzero_index[i][j    ]] = ret4[thread_index][3][0];
				}
				if((j + 1) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]] = ret4[thread_index][0][1];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]] = ret4[thread_index][1][1];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]] = ret4[thread_index][2][1];
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 1]] = ret4[thread_index][3][1];
				}
				if((j + 2) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]] = ret4[thread_index][0][2];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]] = ret4[thread_index][1][2];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]] = ret4[thread_index][2][2];
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 2]] = ret4[thread_index][3][2];
				}
				if((j + 3) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]] = ret4[thread_index][0][3];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]] = ret4[thread_index][1][3];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]] = ret4[thread_index][2][3];
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 3]] = ret4[thread_index][3][3];
				}

				// total_index++;
				total_index[thread_index] += _BNC_D_WIDTH;
			}
			//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);

			//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		}
	} // #pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS][QDSIZE], vb8[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][QDSIZE], ret8[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_qdvector(vec->dim);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		//#pragma omp parallel for private(thread_index, j) ordered
		//for(i = 0; i < mat->row_dim; i++)
		for(i = start_i; i < end_i; i++)
		{
			thread_index = omp_get_thread_num();

			vb8[thread_index][0] = _mm512_set1_pd(vec->element[0][i]);
			vb8[thread_index][1] = _mm512_set1_pd(vec->element[1][i]);
			vb8[thread_index][2] = _mm512_set1_pd(vec->element[2][i]);
			vb8[thread_index][3] = _mm512_set1_pd(vec->element[3][i]);

			//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
			jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//#pragma omp ordered
			for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
			{
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				a8[thread_index][0] = _mm512_load_pd(&(mat->element[0][total_index[thread_index]]));
				a8[thread_index][1] = _mm512_load_pd(&(mat->element[1][total_index[thread_index]]));
				a8[thread_index][2] = _mm512_load_pd(&(mat->element[2][total_index[thread_index]]));
				a8[thread_index][3] = _mm512_load_pd(&(mat->element[3][total_index[thread_index]]));

				ret8[thread_index][0] = _mm512_set_pd(
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]]
				);
				ret8[thread_index][1] = _mm512_set_pd(
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]]
				);
				ret8[thread_index][2] = _mm512_set_pd(
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]]
				);
				ret8[thread_index][3] = _mm512_set_pd(
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j    ]]
				);

				//ret8 = _mm512_fmadd_pd(a8, vb8, ret8);
				_bncavx512_rqd_mul(tmp8[thread_index], a8[thread_index], vb8[thread_index]);

				//#pragma omp critical
				_bncavx512_rqd_add(ret8[thread_index], ret8[thread_index], tmp8[thread_index]);

				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 7]] = ret8[thread_index][0][7];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 6]] = ret8[thread_index][0][6];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 5]] = ret8[thread_index][0][5];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 4]] = ret8[thread_index][0][4];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]] = ret8[thread_index][0][3];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]] = ret8[thread_index][0][2];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]] = ret8[thread_index][0][1];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]] = ret8[thread_index][0][0];

				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 7]] = ret8[thread_index][1][7];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 6]] = ret8[thread_index][1][6];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 5]] = ret8[thread_index][1][5];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 4]] = ret8[thread_index][1][4];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]] = ret8[thread_index][1][3];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]] = ret8[thread_index][1][2];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]] = ret8[thread_index][1][1];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]] = ret8[thread_index][1][0];

				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 7]] = ret8[thread_index][2][7];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 6]] = ret8[thread_index][2][6];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 5]] = ret8[thread_index][2][5];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 4]] = ret8[thread_index][2][4];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]] = ret8[thread_index][2][3];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]] = ret8[thread_index][2][2];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]] = ret8[thread_index][2][1];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]] = ret8[thread_index][2][0];

				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 7]] = ret8[thread_index][3][7];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 6]] = ret8[thread_index][3][6];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 5]] = ret8[thread_index][3][5];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 4]] = ret8[thread_index][3][4];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 3]] = ret8[thread_index][3][3];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 2]] = ret8[thread_index][3][2];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 1]] = ret8[thread_index][3][1];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j    ]] = ret8[thread_index][3][0];

				// total_index++;
				total_index[thread_index] += _BNC_D_WIDTH;
			}
			//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);

			//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP QD A^T*x: per-thread work_vec, gather/FMA/scatter (race-free). */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			work_vec[thread_index] = init_qdvector(vec->dim);
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;

			for(i = start_i; i < end_i; i++)
			{
				svfloat64_t vb0 = svdup_n_f64(vec->element[0][i]);
				svfloat64_t vb1 = svdup_n_f64(vec->element[1][i]);
				svfloat64_t vb2 = svdup_n_f64(vec->element[2][i]);
				svfloat64_t vb3 = svdup_n_f64(vec->element[3][i]);
				long int _ti = 0, _k;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / _vl) * _vl;
				long int jres = nz - jmax;
				double *w0 = work_vec[thread_index]->element[0];
				double *w1 = work_vec[thread_index]->element[1];
				double *w2 = work_vec[thread_index]->element[2];
				double *w3 = work_vec[thread_index]->element[3];
				for(j = 0; j < jmax; j += _vl)
				{
					svbool_t pg = svptrue_b64();
					svfloat64_t a0 = svld1_f64(pg, &(mat->element[0][_ti]));
					svfloat64_t a1 = svld1_f64(pg, &(mat->element[1][_ti]));
					svfloat64_t a2v= svld1_f64(pg, &(mat->element[2][_ti]));
					svfloat64_t a3 = svld1_f64(pg, &(mat->element[3][_ti]));
					svint64_t idx  = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
					svfloat64_t r0 = svld1_gather_s64index_f64(pg, w0, idx);
					svfloat64_t r1 = svld1_gather_s64index_f64(pg, w1, idx);
					svfloat64_t r2 = svld1_gather_s64index_f64(pg, w2, idx);
					svfloat64_t r3 = svld1_gather_s64index_f64(pg, w3, idx);
					svfloat64_t m0, m1, m2, m3;
					_bncsve2_rqd_mul(pg, &m0, &m1, &m2, &m3, a0, a1, a2v, a3, vb0, vb1, vb2, vb3);
					_bncsve2_rqd_add(pg, &r0, &r1, &r2, &r3, r0, r1, r2, r3, m0, m1, m2, m3);
					svst1_scatter_s64index_f64(pg, w0, idx, r0);
					svst1_scatter_s64index_f64(pg, w1, idx, r1);
					svst1_scatter_s64index_f64(pg, w2, idx, r2);
					svst1_scatter_s64index_f64(pg, w3, idx, r3);
					_ti += _vl;
				}
				{
					double mij[QDSIZE], vi[QDSIZE], pr[QDSIZE], cur[QDSIZE];
					long int col_dim = mat->nzero_col_dim[i];
					long int _t;
					vi[0]=vec->element[0][i]; vi[1]=vec->element[1][i]; vi[2]=vec->element[2][i]; vi[3]=vec->element[3][i];
					for(_t = 0; _t < jres; _t++)
					{
						long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
						mij[0]=mat->element[0][_ti]; mij[1]=mat->element[1][_ti]; mij[2]=mat->element[2][_ti]; mij[3]=mat->element[3][_ti];
						cur[0]=w0[idx_t]; cur[1]=w1[idx_t]; cur[2]=w2[idx_t]; cur[3]=w3[idx_t];
						rqd_mul(pr, mij, vi);
						rqd_add(cur, cur, pr);
						w0[idx_t]=cur[0]; w1[idx_t]=cur[1]; w2[idx_t]=cur[2]; w3[idx_t]=cur[3];
						_ti++;
					}
				}
			}
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			work_vec[thread_index] = init_qdvector(vec->dim);
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;

			for(i = start_i; i < end_i; i++)
			{
				float64x2_t vb2[QDSIZE], a2[QDSIZE], r2[QDSIZE], m2[QDSIZE];
				vb2[0]=vdupq_n_f64(vec->element[0][i]); vb2[1]=vdupq_n_f64(vec->element[1][i]); vb2[2]=vdupq_n_f64(vec->element[2][i]); vb2[3]=vdupq_n_f64(vec->element[3][i]);
				long int _ti = 0, _k;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / 2) * 2;
				long int jres = nz - jmax;
				double *w0 = work_vec[thread_index]->element[0];
				double *w1 = work_vec[thread_index]->element[1];
				double *w2 = work_vec[thread_index]->element[2];
				double *w3 = work_vec[thread_index]->element[3];
				double vset[2];
				for(j = 0; j < jmax; j += 2)
				{
					a2[0]=vld1q_f64(&(mat->element[0][_ti])); a2[1]=vld1q_f64(&(mat->element[1][_ti])); a2[2]=vld1q_f64(&(mat->element[2][_ti])); a2[3]=vld1q_f64(&(mat->element[3][_ti]));
					vset[0]=w0[mat->nzero_index[i][j]]; vset[1]=w0[mat->nzero_index[i][j + 1]]; r2[0]=vld1q_f64(vset);
					vset[0]=w1[mat->nzero_index[i][j]]; vset[1]=w1[mat->nzero_index[i][j + 1]]; r2[1]=vld1q_f64(vset);
					vset[0]=w2[mat->nzero_index[i][j]]; vset[1]=w2[mat->nzero_index[i][j + 1]]; r2[2]=vld1q_f64(vset);
					vset[0]=w3[mat->nzero_index[i][j]]; vset[1]=w3[mat->nzero_index[i][j + 1]]; r2[3]=vld1q_f64(vset);
					_bncneon_rqd_mul(m2, a2, vb2);
					_bncneon_rqd_add(r2, r2, m2);
					w0[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[0],0); w1[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[1],0); w2[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[2],0); w3[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[3],0);
					w0[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[0],1); w1[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[1],1); w2[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[2],1); w3[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[3],1);
					_ti += 2;
				}
				{
					double mij[QDSIZE], vi[QDSIZE], pr[QDSIZE], cur[QDSIZE];
					long int col_dim = mat->nzero_col_dim[i];
					long int _t;
					vi[0]=vec->element[0][i]; vi[1]=vec->element[1][i]; vi[2]=vec->element[2][i]; vi[3]=vec->element[3][i];
					for(_t = 0; _t < jres; _t++)
					{
						long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
						mij[0]=mat->element[0][_ti]; mij[1]=mat->element[1][_ti]; mij[2]=mat->element[2][_ti]; mij[3]=mat->element[3][_ti];
						cur[0]=w0[idx_t]; cur[1]=w1[idx_t]; cur[2]=w2[idx_t]; cur[3]=w3[idx_t];
						rqd_mul(pr, mij, vi);
						rqd_add(cur, cur, pr);
						w0[idx_t]=cur[0]; w1[idx_t]=cur[1]; w2[idx_t]=cur[2]; w3[idx_t]=cur[3];
						_ti++;
					}
				}
			}
		}
	}

#else // __AVX2__
	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_qdvector(vec->dim);

        //total_index[thread_index] = 0;

		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		for(i = start_i; i < end_i; i++)
		{
			//tmp_ret[0][i] = (double *)calloc(real_nzero_col_dim[i]);
			//tmp_ret[1][i] = (double *)calloc(real_nzero_col_dim[i]);

			thread_index = omp_get_thread_num();

			vec_i[thread_index][0] = vec->element[0][i];
			vec_i[thread_index][1] = vec->element[1][i];
			vec_i[thread_index][2] = vec->element[2][i];
			vec_i[thread_index][3] = vec->element[3][i];

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//#pragma omp ordered
			//ret_j[thread_index][0] = ret->element[0][i];
			//ret_j[thread_index][1] = ret->element[1][i];
			//rdd_set(ret_j, get_ddvector_i(ret, i));
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{

				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				mat_ji[thread_index][0] = mat->element[0][total_index[thread_index]];
				mat_ji[thread_index][1] = mat->element[1][total_index[thread_index]];
				mat_ji[thread_index][2] = mat->element[2][total_index[thread_index]];
				mat_ji[thread_index][3] = mat->element[3][total_index[thread_index]];
				//ret_j[thread_index][0] = ret->element[0][mat->nzero_index[i][j]];
				//ret_j[thread_index][1] = ret->element[1][mat->nzero_index[i][j]];
				ret_j[thread_index][0] = work_vec[thread_index]->element[0][mat->nzero_index[i][j]];
				ret_j[thread_index][1] = work_vec[thread_index]->element[1][mat->nzero_index[i][j]];
				ret_j[thread_index][2] = work_vec[thread_index]->element[2][mat->nzero_index[i][j]];
				ret_j[thread_index][3] = work_vec[thread_index]->element[3][mat->nzero_index[i][j]];
				//rdd_mul(tmp, mat->element[total_index]), get_ddvector_i(vec, i));
				rqd_mul(tmp[thread_index], mat_ji[thread_index], vec_i[thread_index]);
				//rdd_add(ret->element[mat->nzero_index[i][j]], ret->element[mat->nzero_index[i][j]], tmp);

				//#pragma omp critical
				//{
				rqd_add(ret_j[thread_index], ret_j[thread_index], tmp[thread_index]);
				//set_ddvector_i(word, mat->nzero_index[i][j], ret_j[thread_index]);
				set_qdvector_i(work_vec[thread_index], mat->nzero_index[i][j], ret_j[thread_index]);
				//}
				total_index[thread_index]++;
			}
			//set_ddvector_i(ret, i, ret_j[thread_index]);
		}
	} // #omp parallel private(thread_index, num_threads, i, j)

#endif // __AVX2__

    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
		#pragma omp critical
		{
			add_qdvector(ret, ret, work_vec[thread_index]);
			free_qdvector(work_vec[thread_index]);
		}
	}

	return SUCCESS;
}

/* Multiply DRSMatrix^T * QDVector */
int _bncomp_mul_drsmatrixt_qdvec(QDVector ret, DRSMatrix mat, QDVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	//ddfloat tmp, mat_ji, vec_j, ret_j;
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE], mat_ji[BNCOMP_MAX_NUM_THREADS], vec_i[BNCOMP_MAX_NUM_THREADS][QDSIZE], ret_j[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	QDVector work_vec[BNCOMP_MAX_NUM_THREADS];
    int thread_index, num_threads;
	long int div_row, start_i, end_i;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//for(i = 0; i < mat->row_dim; i++)
	//	mpf_set_ui(get_mpfvector_i(ret, i), 0UL);

	// ret := 0
	set0_qdvector(ret);
	//total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS], vb4[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][QDSIZE], ret4[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];
	double vset[BNCOMP_MAX_NUM_THREADS][_BNC_D_WIDTH];

	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_qdvector(vec->dim);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		//for(i = 0; i < mat->row_dim; i++)
		for(i = start_i; i < end_i; i++)
		{
			thread_index = omp_get_thread_num();

			vb4[thread_index][0] = _mm256_set1_pd(vec->element[0][i]);
			vb4[thread_index][1] = _mm256_set1_pd(vec->element[1][i]);
			vb4[thread_index][2] = _mm256_set1_pd(vec->element[2][i]);
			vb4[thread_index][3] = _mm256_set1_pd(vec->element[3][i]);

			//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
			jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//jmax = (mat->row_dim / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->row_dim % _BNC_D_WIDTH;
			//#pragma omp ordered
			for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
			{
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				a4[thread_index] = _mm256_load_pd(&(mat->element[total_index[thread_index]]));
				//a4[thread_index][1] = _mm256_load_pd(&(mat->element[1][total_index[thread_index]]));
				//a4[thread_index][2] = _mm256_load_pd(&(mat->element[2][total_index[thread_index]]));
				//a4[thread_index][3] = _mm256_load_pd(&(mat->element[3][total_index[thread_index]]));

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]];
				ret4[thread_index][0] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[0][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[0][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[0][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[0][mat->nzero_index[i][j    ]]
				);

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]];
				ret4[thread_index][1] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[1][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[1][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[1][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[1][mat->nzero_index[i][j    ]]
				);

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]];
				ret4[thread_index][2] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[1][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[1][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[1][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[1][mat->nzero_index[i][j    ]]
				);

				vset[thread_index][0] = 0.0; vset[thread_index][1] = 0.0; vset[thread_index][2] = 0.0; vset[thread_index][3] = 0.0;
				if((j    ) < mat->nzero_col_dim[i]) vset[thread_index][0] = work_vec[thread_index]->element[3][mat->nzero_index[i][j    ]];
				if((j + 1) < mat->nzero_col_dim[i]) vset[thread_index][1] = work_vec[thread_index]->element[3][mat->nzero_index[i][j + 1]];
				if((j + 2) < mat->nzero_col_dim[i]) vset[thread_index][2] = work_vec[thread_index]->element[3][mat->nzero_index[i][j + 2]];
				if((j + 3) < mat->nzero_col_dim[i]) vset[thread_index][3] = work_vec[thread_index]->element[3][mat->nzero_index[i][j + 3]];
				ret4[thread_index][3] = _mm256_set_pd(
					vset[thread_index][3], // ret->element[1][mat->nzero_index[i][j + 3]],
					vset[thread_index][2], // ret->element[1][mat->nzero_index[i][j + 2]],
					vset[thread_index][1], // ret->element[1][mat->nzero_index[i][j + 1]],
					vset[thread_index][0]  // ret->element[1][mat->nzero_index[i][j    ]]
				);
				//ret4 = _mm256_fmadd_pd(a4, vb4, ret4);
				//_bncavx2_rqd_mul(tmp4[thread_index], a4[thread_index], vb4[thread_index]);
				_bncavx2_rqd_mul_d(tmp4[thread_index], vb4[thread_index], a4[thread_index]);

			// #pragma omp critical
				_bncavx2_rqd_add(ret4[thread_index], ret4[thread_index], tmp4[thread_index]);

				if((j    ) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]] = ret4[thread_index][0][0];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]] = ret4[thread_index][1][0];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]] = ret4[thread_index][2][0];
					work_vec[thread_index]->element[3][mat->nzero_index[i][j    ]] = ret4[thread_index][3][0];
				}
				if((j + 1) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]] = ret4[thread_index][0][1];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]] = ret4[thread_index][1][1];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]] = ret4[thread_index][2][1];
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 1]] = ret4[thread_index][3][1];
				}
				if((j + 2) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]] = ret4[thread_index][0][2];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]] = ret4[thread_index][1][2];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]] = ret4[thread_index][2][2];
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 2]] = ret4[thread_index][3][2];
				}
				if((j + 3) < mat->nzero_col_dim[i])
				{
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]] = ret4[thread_index][0][3];
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]] = ret4[thread_index][1][3];
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]] = ret4[thread_index][2][3];
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 3]] = ret4[thread_index][3][3];
				}

				// total_index++;
				total_index[thread_index] += _BNC_D_WIDTH;
			}
			//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);

			//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		}
	} // #pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS], vb8[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][QDSIZE], ret8[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	long int jmax[BNCOMP_MAX_NUM_THREADS], jres[BNCOMP_MAX_NUM_THREADS], col_dim[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_qdvector(vec->dim);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		//#pragma omp parallel for private(thread_index, j) ordered
		//for(i = 0; i < mat->row_dim; i++)
		for(i = start_i; i < end_i; i++)
		{
			thread_index = omp_get_thread_num();

			vb8[thread_index][0] = _mm512_set1_pd(vec->element[0][i]);
			vb8[thread_index][1] = _mm512_set1_pd(vec->element[1][i]);
			vb8[thread_index][2] = _mm512_set1_pd(vec->element[2][i]);
			vb8[thread_index][3] = _mm512_set1_pd(vec->element[3][i]);

			//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
			jmax[thread_index] = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
			jres[thread_index] =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//#pragma omp ordered
			for(j = 0; j < jmax[thread_index]; j += _BNC_D_WIDTH)
			{
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				a8[thread_index] = _mm512_load_pd(&(mat->element[total_index[thread_index]]));
				//a8[thread_index][1] = _mm512_load_pd(&(mat->element[1][total_index[thread_index]]));
				//a8[thread_index][2] = _mm512_load_pd(&(mat->element[2][total_index[thread_index]]));
				//a8[thread_index][3] = _mm512_load_pd(&(mat->element[3][total_index[thread_index]]));

				ret8[thread_index][0] = _mm512_set_pd(
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]]
				);
				ret8[thread_index][1] = _mm512_set_pd(
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]]
				);
				ret8[thread_index][2] = _mm512_set_pd(
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]]
				);
				ret8[thread_index][3] = _mm512_set_pd(
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 7]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 6]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 5]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 4]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 3]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 2]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j + 1]],
					work_vec[thread_index]->element[3][mat->nzero_index[i][j    ]]
				);

				//ret8 = _mm512_fmadd_pd(a8, vb8, ret8);
				//_bncavx512_rqd_mul(tmp8[thread_index], a8[thread_index], vb8[thread_index]);
				_bncavx512_rqd_mul_d(tmp8[thread_index], vb8[thread_index], a8[thread_index]);

				//#pragma omp critical
				_bncavx512_rqd_add(ret8[thread_index], ret8[thread_index], tmp8[thread_index]);

				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 7]] = ret8[thread_index][0][7];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 6]] = ret8[thread_index][0][6];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 5]] = ret8[thread_index][0][5];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 4]] = ret8[thread_index][0][4];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 3]] = ret8[thread_index][0][3];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 2]] = ret8[thread_index][0][2];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j + 1]] = ret8[thread_index][0][1];
				work_vec[thread_index]->element[0][mat->nzero_index[i][j    ]] = ret8[thread_index][0][0];

				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 7]] = ret8[thread_index][1][7];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 6]] = ret8[thread_index][1][6];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 5]] = ret8[thread_index][1][5];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 4]] = ret8[thread_index][1][4];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 3]] = ret8[thread_index][1][3];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 2]] = ret8[thread_index][1][2];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j + 1]] = ret8[thread_index][1][1];
				work_vec[thread_index]->element[1][mat->nzero_index[i][j    ]] = ret8[thread_index][1][0];

				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 7]] = ret8[thread_index][2][7];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 6]] = ret8[thread_index][2][6];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 5]] = ret8[thread_index][2][5];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 4]] = ret8[thread_index][2][4];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 3]] = ret8[thread_index][2][3];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 2]] = ret8[thread_index][2][2];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j + 1]] = ret8[thread_index][2][1];
				work_vec[thread_index]->element[2][mat->nzero_index[i][j    ]] = ret8[thread_index][2][0];

				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 7]] = ret8[thread_index][3][7];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 6]] = ret8[thread_index][3][6];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 5]] = ret8[thread_index][3][5];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 4]] = ret8[thread_index][3][4];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 3]] = ret8[thread_index][3][3];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 2]] = ret8[thread_index][3][2];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j + 1]] = ret8[thread_index][3][1];
				work_vec[thread_index]->element[3][mat->nzero_index[i][j    ]] = ret8[thread_index][3][0];

				// total_index++;
				total_index[thread_index] += _BNC_D_WIDTH;
			}
			//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);

			//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (OpenMP)
	/* SVE2 + OpenMP D^T*QD: mat plain double promoted to QD; per-thread work_vec. */
	{
		long _vl = (long)svcntd();
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			work_vec[thread_index] = init_qdvector(vec->dim);
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;

			for(i = start_i; i < end_i; i++)
			{
				svfloat64_t vb0 = svdup_n_f64(vec->element[0][i]);
				svfloat64_t vb1 = svdup_n_f64(vec->element[1][i]);
				svfloat64_t vb2 = svdup_n_f64(vec->element[2][i]);
				svfloat64_t vb3 = svdup_n_f64(vec->element[3][i]);
				svfloat64_t zerov = svdup_n_f64(0.0);
				long int _ti = 0, _k;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / _vl) * _vl;
				long int jres = nz - jmax;
				double *w0 = work_vec[thread_index]->element[0];
				double *w1 = work_vec[thread_index]->element[1];
				double *w2 = work_vec[thread_index]->element[2];
				double *w3 = work_vec[thread_index]->element[3];
				for(j = 0; j < jmax; j += _vl)
				{
					svbool_t pg = svptrue_b64();
					svfloat64_t a_v = svld1_f64(pg, &(mat->element[_ti]));
					svint64_t idx  = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
					svfloat64_t r0 = svld1_gather_s64index_f64(pg, w0, idx);
					svfloat64_t r1 = svld1_gather_s64index_f64(pg, w1, idx);
					svfloat64_t r2 = svld1_gather_s64index_f64(pg, w2, idx);
					svfloat64_t r3 = svld1_gather_s64index_f64(pg, w3, idx);
					svfloat64_t m0, m1, m2, m3;
					_bncsve2_rqd_mul(pg, &m0, &m1, &m2, &m3, a_v, zerov, zerov, zerov, vb0, vb1, vb2, vb3);
					_bncsve2_rqd_add(pg, &r0, &r1, &r2, &r3, r0, r1, r2, r3, m0, m1, m2, m3);
					svst1_scatter_s64index_f64(pg, w0, idx, r0);
					svst1_scatter_s64index_f64(pg, w1, idx, r1);
					svst1_scatter_s64index_f64(pg, w2, idx, r2);
					svst1_scatter_s64index_f64(pg, w3, idx, r3);
					_ti += _vl;
				}
				{
					double mij[QDSIZE], vi[QDSIZE], pr[QDSIZE], cur[QDSIZE];
					long int col_dim = mat->nzero_col_dim[i];
					long int _t;
					vi[0]=vec->element[0][i]; vi[1]=vec->element[1][i]; vi[2]=vec->element[2][i]; vi[3]=vec->element[3][i];
					for(_t = 0; _t < jres; _t++)
					{
						long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
						mij[0]=mat->element[_ti]; mij[1]=0.0; mij[2]=0.0; mij[3]=0.0;
						cur[0]=w0[idx_t]; cur[1]=w1[idx_t]; cur[2]=w2[idx_t]; cur[3]=w3[idx_t];
						rqd_mul(pr, mij, vi);
						rqd_add(cur, cur, pr);
						w0[idx_t]=cur[0]; w1[idx_t]=cur[1]; w2[idx_t]=cur[2]; w3[idx_t]=cur[3];
						_ti++;
					}
				}
			}
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // Arm NEON (OpenMP)
	{
		#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
		{
			thread_index = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			work_vec[thread_index] = init_qdvector(vec->dim);
			div_row = mat->row_dim / num_threads;
			if(mat->row_dim % num_threads != 0) div_row++;
			start_i = thread_index * div_row;
			end_i = start_i + div_row;
			if(end_i > mat->row_dim) end_i = mat->row_dim;

			for(i = start_i; i < end_i; i++)
			{
				float64x2_t vb2[QDSIZE], r2[QDSIZE], m2[QDSIZE], a_qd[QDSIZE];
				vb2[0]=vdupq_n_f64(vec->element[0][i]); vb2[1]=vdupq_n_f64(vec->element[1][i]); vb2[2]=vdupq_n_f64(vec->element[2][i]); vb2[3]=vdupq_n_f64(vec->element[3][i]);
				a_qd[1]=vdupq_n_f64(0.0); a_qd[2]=vdupq_n_f64(0.0); a_qd[3]=vdupq_n_f64(0.0);
				long int _ti = 0, _k;
				for(_k = 0; _k < i; _k++) _ti += mat->real_nzero_col_dim[_k];
				long int nz = mat->nzero_col_dim[i];
				long int jmax = (nz / 2) * 2;
				long int jres = nz - jmax;
				double *w0 = work_vec[thread_index]->element[0];
				double *w1 = work_vec[thread_index]->element[1];
				double *w2 = work_vec[thread_index]->element[2];
				double *w3 = work_vec[thread_index]->element[3];
				double vset[2];
				for(j = 0; j < jmax; j += 2)
				{
					a_qd[0] = vld1q_f64(&(mat->element[_ti]));
					vset[0]=w0[mat->nzero_index[i][j]]; vset[1]=w0[mat->nzero_index[i][j + 1]]; r2[0]=vld1q_f64(vset);
					vset[0]=w1[mat->nzero_index[i][j]]; vset[1]=w1[mat->nzero_index[i][j + 1]]; r2[1]=vld1q_f64(vset);
					vset[0]=w2[mat->nzero_index[i][j]]; vset[1]=w2[mat->nzero_index[i][j + 1]]; r2[2]=vld1q_f64(vset);
					vset[0]=w3[mat->nzero_index[i][j]]; vset[1]=w3[mat->nzero_index[i][j + 1]]; r2[3]=vld1q_f64(vset);
					_bncneon_rqd_mul(m2, a_qd, vb2);
					_bncneon_rqd_add(r2, r2, m2);
					w0[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[0],0); w1[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[1],0); w2[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[2],0); w3[mat->nzero_index[i][j]]=vgetq_lane_f64(r2[3],0);
					w0[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[0],1); w1[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[1],1); w2[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[2],1); w3[mat->nzero_index[i][j + 1]]=vgetq_lane_f64(r2[3],1);
					_ti += 2;
				}
				{
					double mij[QDSIZE], vi[QDSIZE], pr[QDSIZE], cur[QDSIZE];
					long int col_dim = mat->nzero_col_dim[i];
					long int _t;
					vi[0]=vec->element[0][i]; vi[1]=vec->element[1][i]; vi[2]=vec->element[2][i]; vi[3]=vec->element[3][i];
					for(_t = 0; _t < jres; _t++)
					{
						long int idx_t = mat->nzero_index[i][col_dim - jres + _t];
						mij[0]=mat->element[_ti]; mij[1]=0.0; mij[2]=0.0; mij[3]=0.0;
						cur[0]=w0[idx_t]; cur[1]=w1[idx_t]; cur[2]=w2[idx_t]; cur[3]=w3[idx_t];
						rqd_mul(pr, mij, vi);
						rqd_add(cur, cur, pr);
						w0[idx_t]=cur[0]; w1[idx_t]=cur[1]; w2[idx_t]=cur[2]; w3[idx_t]=cur[3];
						_ti++;
					}
				}
			}
		}
	}

#else // __AVX2__
	#pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

		work_vec[thread_index] = init_qdvector(vec->dim);

        //total_index[thread_index] = 0;

		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		for(i = start_i; i < end_i; i++)
		{
			//tmp_ret[0][i] = (double *)calloc(real_nzero_col_dim[i]);
			//tmp_ret[1][i] = (double *)calloc(real_nzero_col_dim[i]);

			thread_index = omp_get_thread_num();

			vec_i[thread_index][0] = vec->element[0][i];
			vec_i[thread_index][1] = vec->element[1][i];
			vec_i[thread_index][2] = vec->element[2][i];
			vec_i[thread_index][3] = vec->element[3][i];

			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j]; // _BNC_D_WIDTH;

			//#pragma omp ordered
			//ret_j[thread_index][0] = ret->element[0][i];
			//ret_j[thread_index][1] = ret->element[1][i];
			//rdd_set(ret_j, get_ddvector_i(ret, i));
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{

				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				mat_ji[thread_index] = mat->element[total_index[thread_index]];
				//mat_ji[thread_index][1] = mat->element[1][total_index[thread_index]];
				//mat_ji[thread_index][2] = mat->element[2][total_index[thread_index]];
				//mat_ji[thread_index][3] = mat->element[3][total_index[thread_index]];
				//ret_j[thread_index][0] = ret->element[0][mat->nzero_index[i][j]];
				//ret_j[thread_index][1] = ret->element[1][mat->nzero_index[i][j]];
				ret_j[thread_index][0] = work_vec[thread_index]->element[0][mat->nzero_index[i][j]];
				ret_j[thread_index][1] = work_vec[thread_index]->element[1][mat->nzero_index[i][j]];
				ret_j[thread_index][2] = work_vec[thread_index]->element[2][mat->nzero_index[i][j]];
				ret_j[thread_index][3] = work_vec[thread_index]->element[3][mat->nzero_index[i][j]];
				//rdd_mul(tmp, mat->element[total_index]), get_ddvector_i(vec, i));
				//rqd_mul(tmp[thread_index], mat_ji[thread_index], vec_i[thread_index]);
				rqd_mul_d(tmp[thread_index], vec_i[thread_index], mat_ji[thread_index]);
				//rdd_add(ret->element[mat->nzero_index[i][j]], ret->element[mat->nzero_index[i][j]], tmp);

				//#pragma omp critical
				//{
				rqd_add(ret_j[thread_index], ret_j[thread_index], tmp[thread_index]);
				//set_ddvector_i(word, mat->nzero_index[i][j], ret_j[thread_index]);
				set_qdvector_i(work_vec[thread_index], mat->nzero_index[i][j], ret_j[thread_index]);
				//}
				total_index[thread_index]++;
			}
			//set_ddvector_i(ret, i, ret_j[thread_index]);
		}
	} // #omp parallel private(thread_index, num_threads, i, j)

#endif // __AVX2__

    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
		#pragma omp critical
		{
			add_qdvector(ret, ret, work_vec[thread_index]);
			free_qdvector(work_vec[thread_index]);
		}
	}

	return SUCCESS;
}

/* Multiply CQDRSMatrix * CQDVector */
int _bncomp_mul_cqdrsmatrix_cqdvec(CQDVector ret, CQDRSMatrix mat, CQDVector vec)
{
	long int i, j, total_index;
	QDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_qdvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_qdrsmatrix_qdvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_qdrsmatrix_qdvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_qdrsmatrix_qdvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_qdrsmatrix_qdvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_sub_qdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_qdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_qdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CDRSMatrix * CQDVector */
int _bncomp_mul_cdrsmatrix_cqdvec(CQDVector ret, CDRSMatrix mat, CQDVector vec)
{
	long int i, j, total_index;
	QDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_qdvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_drsmatrix_qdvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_drsmatrix_qdvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_drsmatrix_qdvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_drsmatrix_qdvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_sub_qdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_qdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_qdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CQDRSMatrix^T * CQDVector */
int _bncomp_mul_cqdrsmatrixt_cqdvec(CQDVector ret, CQDRSMatrix mat, CQDVector vec)
{
	long int i, j, total_index;
	QDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_qdvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_qdrsmatrixt_qdvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_qdrsmatrixt_qdvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_qdrsmatrixt_qdvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_qdrsmatrixt_qdvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_sub_qdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_qdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_qdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CDRSMatrix^T * CQDVector */
int _bncomp_mul_cdrsmatrixt_cqdvec(CQDVector ret, CDRSMatrix mat, CQDVector vec)
{
	long int i, j, total_index;
	QDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_qdvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_drsmatrixt_qdvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_drsmatrixt_qdvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_drsmatrixt_qdvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_drsmatrixt_qdvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_sub_qdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_add_qdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_qdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply conj(CQDRSMatrix)^T * CQDVector */
int _bncomp_mul_cqdrsmatrixs_cqdvec(CQDVector ret, CQDRSMatrix mat, CQDVector vec)
{
	long int i, j, total_index;
	QDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_qdvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_qdrsmatrixt_qdvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_qdrsmatrixt_qdvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_qdrsmatrixt_qdvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_qdrsmatrixt_qdvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_add_qdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_sub_qdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_qdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply conj(CDRSMatrix)^T * CQDVector */
int _bncomp_mul_cdrsmatrixs_cqdvec(CQDVector ret, CDRSMatrix mat, CQDVector vec)
{
	long int i, j, total_index;
	QDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_qdvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	_bncomp_mul_drsmatrixt_qdvec(tmp_vec[0], mat->re, vec->re);
	_bncomp_mul_drsmatrixt_qdvec(tmp_vec[1], mat->re, vec->im);
	_bncomp_mul_drsmatrixt_qdvec(tmp_vec[2], mat->im, vec->re);
	_bncomp_mul_drsmatrixt_qdvec(tmp_vec[3], mat->im, vec->im);

	_bncomp_add_qdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	_bncomp_sub_qdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_qdvector(tmp_vec[i]);

	return SUCCESS;
}

//-----------------------------------------------
// MPF precision
//-----------------------------------------------
#ifdef USE_GMP

/* Multiply MPFRSMatrix * MPFVector */
int _bncomp_mul_mpfrsmatrix_mpfvec(MPFVector ret, MPFRSMatrix mat, MPFVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	unsigned long prec;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
    int thread_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
    	mpf_init2(tmp[thread_index], prec);

        //total_index[thread_index] = 0;
    }

	//total_index = 0;
    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

        // calculate accmulated total_index
        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->nzero_col_dim[j];

		//get_mpfvector_i(ret, i) = 0.0;
		mpf_set_ui(get_mpfvector_i(ret, i), 0UL);
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpf_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mpf_mul(tmp[thread_index], (mpf_ptr)(mat->element[total_index[thread_index]]), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mpf_add(get_mpfvector_i(ret, i), get_mpfvector_i(ret, i), tmp[thread_index]);

    		total_index[thread_index]++;
		}
	}

    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
    	mpf_clear(tmp[thread_index]);
    }

	return SUCCESS;
}

/* Multiply DRSMatrix * MPFVector */
int _bncomp_mul_drsmatrix_mpfvec(MPFVector ret, DRSMatrix mat, MPFVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	unsigned long prec;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
    int thread_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
    	mpf_init2(tmp[thread_index], prec);

        //total_index[thread_index] = 0;
    }

	//total_index = 0;
    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

        // calculate accmulated total_index
        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->real_nzero_col_dim[j];

		//get_mpfvector_i(ret, i) = 0.0;
		mpf_set_ui(get_mpfvector_i(ret, i), 0UL);
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpf_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			//mpf_mul(tmp[thread_index], (mpf_ptr)(mat->element[total_index[thread_index]]), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mpf_mul_d(tmp[thread_index], get_mpfvector_i(vec, mat->nzero_index[i][j]), mat->element[total_index[thread_index]]);

			mpf_add(get_mpfvector_i(ret, i), get_mpfvector_i(ret, i), tmp[thread_index]);

    		total_index[thread_index]++;
		}
    	total_index[thread_index] += (mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]);		
	}

    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
    	mpf_clear(tmp[thread_index]);
    }

	return SUCCESS;
}


/* Multiply MPFRSMatrix^T * MPFVector */
int _bncomp_mul_mpfrsmatrixt_mpfvec(MPFVector ret, MPFRSMatrix mat, MPFVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	unsigned long prec;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS], ret_tmp[BNCOMP_MAX_NUM_THREADS], vi[BNCOMP_MAX_NUM_THREADS];
	MPFVector work_vec[BNCOMP_MAX_NUM_THREADS];
    int thread_index, num_threads;
	long int div_row, start_i, end_i;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//prec = ret->prec;
	//mpf_init2(tmp, prec);
	prec = ret->prec;
    #pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

    	mpf_init2(tmp[thread_index], prec);
    	mpf_init2(vi[thread_index], prec);
		work_vec[thread_index] = init2_mpfvector(vec->dim, prec);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		for(i = start_i; i < end_i; i++)
		{
			// calculate accmulated total_index
			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->nzero_col_dim[j];

			//#pragma omp section
			//#pragma omp critical			
			mpf_set(vi[thread_index], get_mpfvector_i(vec, i));
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{

		//#ifndef USE_MPFR
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				mpf_mul(tmp[thread_index], (mpf_ptr)(mat->element[total_index[thread_index]]), vi[thread_index]); // get_mpfvector_i(vec, i));

				//#pragma omp critical
				mpf_add((mpf_ptr)(work_vec[thread_index]->element[mat->nzero_index[i][j]]), (mpf_ptr)(work_vec[thread_index]->element[mat->nzero_index[i][j]]), tmp[thread_index]);
				//mpf_add(ret_tmp[thread_index], ret_tmp[thread_index], tmp[thread_index]);
		//#else
		/*
				//mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j), tmp[thread_index], bnc_default_rounding_mode);
				//#pragma omp critical
				mpfr_fma((mpf_ptr)(ret->element[mat->nzero_index[i][j]]), (mpf_ptr)(mat->element[total_index[thread_index]]), get_mpfvector_i(vec, i), (mpf_ptr)(ret->element[mat->nzero_index[i][j]]), MPFR_RNDN);
		*/
		//#endif
				total_index[thread_index]++;
			}
			//set_mpfvector_i(ret, i, ret_tmp[thread_index]);	
		}

    	mpf_clear(tmp[thread_index]);
		mpf_clear(vi[thread_index]);
    }

	set0_mpfvector(ret);
    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
		#pragma omp critical
		{
			add_mpfvector(ret, ret, work_vec[thread_index]);
			free_mpfvector(work_vec[thread_index]);
		}
	}

	return SUCCESS;
}

/* Multiply DRSMatrix^T * MPFVector */
int _bncomp_mul_drsmatrixt_mpfvec(MPFVector ret, DRSMatrix mat, MPFVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	unsigned long prec;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS], ret_tmp[BNCOMP_MAX_NUM_THREADS], vi[BNCOMP_MAX_NUM_THREADS];
	MPFVector work_vec[BNCOMP_MAX_NUM_THREADS];
    int thread_index, num_threads;
	long int div_row, start_i, end_i;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//prec = ret->prec;
	//mpf_init2(tmp, prec);
	prec = ret->prec;
    #pragma omp parallel  private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

    	mpf_init2(tmp[thread_index], prec);
    	mpf_init2(vi[thread_index], prec);
		work_vec[thread_index] = init2_mpfvector(vec->dim, prec);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		for(i = start_i; i < end_i; i++)
		{
			// calculate accmulated total_index
			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->real_nzero_col_dim[j];

			//#pragma omp section
			//#pragma omp critical			
			mpf_set(vi[thread_index], get_mpfvector_i(vec, i));
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{

		//#ifndef USE_MPFR
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				//mpf_mul(tmp[thread_index], (mpf_ptr)(mat->element[total_index[thread_index]]), vi[thread_index]); // get_mpfvector_i(vec, i));
				mpf_mul_d(tmp[thread_index], vi[thread_index], mat->element[total_index[thread_index]]); 

				//#pragma omp critical
				mpf_add((mpf_ptr)(work_vec[thread_index]->element[mat->nzero_index[i][j]]), (mpf_ptr)(work_vec[thread_index]->element[mat->nzero_index[i][j]]), tmp[thread_index]);
				//mpf_add(ret_tmp[thread_index], ret_tmp[thread_index], tmp[thread_index]);
		//#else
		/*
				//mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j), tmp[thread_index], bnc_default_rounding_mode);
				//#pragma omp critical
				mpfr_fma((mpf_ptr)(ret->element[mat->nzero_index[i][j]]), (mpf_ptr)(mat->element[total_index[thread_index]]), get_mpfvector_i(vec, i), (mpf_ptr)(ret->element[mat->nzero_index[i][j]]), MPFR_RNDN);
		*/
		//#endif
				total_index[thread_index]++;
			}
	    	total_index[thread_index] += (mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]);		
			//set_mpfvector_i(ret, i, ret_tmp[thread_index]);	
		}

    	mpf_clear(tmp[thread_index]);
		mpf_clear(vi[thread_index]);
    }

	set0_mpfvector(ret);
    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
		#pragma omp critical
		{
			add_mpfvector(ret, ret, work_vec[thread_index]);
			free_mpfvector(work_vec[thread_index]);
		}
	}

	return SUCCESS;
}

/* Multiply CMPFRSMatrix * CMPFVector */
int _bncomp_mul_cmpfrsmatrix_cmpfvec(CMPFVector ret, CMPFRSMatrix mat, CMPFVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	unsigned long prec;
	mpc_t tmp[BNCOMP_MAX_NUM_THREADS];
    int thread_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
    	mpc_init2(tmp[thread_index], prec);
    }
	//mpc_init2(tmp, prec);

	//total_index = 0;
    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

        // calculate accmulated total_index
        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->nzero_col_dim[j];

		mpc_set_si(get_cmpfvector_i(ret, i), 0UL, MPC_RNDNN);
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpc_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpc_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mpc_mul(tmp[thread_index], (mpc_ptr)(mat->element[total_index[thread_index]]), get_cmpfvector_i(vec, mat->nzero_index[i][j]), MPC_RNDNN);
			mpc_add(get_cmpfvector_i(ret, i), get_cmpfvector_i(ret, i), tmp[thread_index], MPC_RNDNN);
			total_index[thread_index]++;
		}
	}

	//mpf_clear(tmp);
    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
    	mpc_clear(tmp[thread_index]);
    }
	//mpc_clear(tmp);

	return SUCCESS;
}

/* Multiply CMPFRSMatrix^T * MPFVector */
int _bncomp_mul_cmpfrsmatrixt_cmpfvec(CMPFVector ret, CMPFRSMatrix mat, CMPFVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	unsigned long prec;
	mpc_t tmp[BNCOMP_MAX_NUM_THREADS], vi[BNCOMP_MAX_NUM_THREADS];
	CMPFVector work_vec[BNCOMP_MAX_NUM_THREADS];
    int thread_index, num_threads;
	long int div_row, start_i, end_i;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//prec = ret->prec;
	//mpf_init2(tmp, prec);
	prec = ret->prec;
    #pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

    	mpc_init2(tmp[thread_index], prec);
    	mpc_init2(vi[thread_index], prec);
		work_vec[thread_index] = init2_cmpfvector(vec->dim, prec);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		for(i = start_i; i < end_i; i++)
		{
			// calculate accmulated total_index
			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->nzero_col_dim[j];

			//#pragma omp section
			//#pragma omp critical			
			mpc_set(vi[thread_index], get_cmpfvector_i(vec, i), MPC_RNDNN);
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{

		//#ifndef USE_MPFR
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				mpc_mul(tmp[thread_index], (mpc_ptr)(mat->element[total_index[thread_index]]), vi[thread_index], MPC_RNDNN); // get_mpfvector_i(vec, i));

				//#pragma omp critical
				mpc_add((mpc_ptr)(work_vec[thread_index]->element[mat->nzero_index[i][j]]), (mpc_ptr)(work_vec[thread_index]->element[mat->nzero_index[i][j]]), tmp[thread_index], MPC_RNDNN);
				//mpf_add(ret_tmp[thread_index], ret_tmp[thread_index], tmp[thread_index]);
		//#else
		/*
				//mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j), tmp[thread_index], bnc_default_rounding_mode);
				//#pragma omp critical
				mpfr_fma((mpf_ptr)(ret->element[mat->nzero_index[i][j]]), (mpf_ptr)(mat->element[total_index[thread_index]]), get_mpfvector_i(vec, i), (mpf_ptr)(ret->element[mat->nzero_index[i][j]]), MPFR_RNDN);
		*/
		//#endif
				total_index[thread_index]++;
			}
			//set_mpfvector_i(ret, i, ret_tmp[thread_index]);	
		}

    	mpc_clear(tmp[thread_index]);
		mpc_clear(vi[thread_index]);
    }

	set0_cmpfvector(ret);
    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
		#pragma omp critical
		{
			add_cmpfvector(ret, ret, work_vec[thread_index]);
			free_cmpfvector(work_vec[thread_index]);
		}
	}

	return SUCCESS;
}

/* Multiply conj(CMPFRSMatrix)^T * MPFVector */
int _bncomp_mul_cmpfrsmatrixs_cmpfvec(CMPFVector ret, CMPFRSMatrix mat, CMPFVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	unsigned long prec;
	mpc_t tmp[BNCOMP_MAX_NUM_THREADS], mat_ij_conj[BNCOMP_MAX_NUM_THREADS], vi[BNCOMP_MAX_NUM_THREADS];
	CMPFVector work_vec[BNCOMP_MAX_NUM_THREADS];
    int thread_index, num_threads;
	long int div_row, start_i, end_i;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//prec = ret->prec;
	//mpf_init2(tmp, prec);
	prec = ret->prec;
    #pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

    	mpc_init2(tmp[thread_index], prec);
    	mpc_init2(vi[thread_index], prec);
		mpc_init2(mat_ij_conj[thread_index], prec);
		work_vec[thread_index] = init2_cmpfvector(vec->dim, prec);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->row_dim / num_threads;
		if(mat->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->row_dim) end_i = mat->row_dim;

		for(i = start_i; i < end_i; i++)
		{
			// calculate accmulated total_index
			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->nzero_col_dim[j];

			//#pragma omp section
			//#pragma omp critical			
			mpc_set(vi[thread_index], get_cmpfvector_i(vec, i), MPC_RNDNN);
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{

		//#ifndef USE_MPFR
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				mpc_conj(mat_ij_conj[thread_index], (mpc_ptr)(mat->element[total_index[thread_index]]), MPC_RNDNN);
				//mpc_mul(tmp[thread_index], (mpc_ptr)(mat->element[total_index[thread_index]]), vi[thread_index], MPC_RNDNN); // get_mpfvector_i(vec, i));
				mpc_mul(tmp[thread_index], mat_ij_conj[thread_index], vi[thread_index], MPC_RNDNN); // get_mpfvector_i(vec, i));


				//#pragma omp critical
				mpc_add((mpc_ptr)(work_vec[thread_index]->element[mat->nzero_index[i][j]]), (mpc_ptr)(work_vec[thread_index]->element[mat->nzero_index[i][j]]), tmp[thread_index], MPC_RNDNN);
				//mpf_add(ret_tmp[thread_index], ret_tmp[thread_index], tmp[thread_index]);
		//#else
		/*
				//mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j), tmp[thread_index], bnc_default_rounding_mode);
				//#pragma omp critical
				mpfr_fma((mpf_ptr)(ret->element[mat->nzero_index[i][j]]), (mpf_ptr)(mat->element[total_index[thread_index]]), get_mpfvector_i(vec, i), (mpf_ptr)(ret->element[mat->nzero_index[i][j]]), MPFR_RNDN);
		*/
		//#endif
				total_index[thread_index]++;
			}
			//set_mpfvector_i(ret, i, ret_tmp[thread_index]);	
		}

    	mpc_clear(tmp[thread_index]);
		mpc_clear(vi[thread_index]);
		mpc_clear(mat_ij_conj[thread_index]);
    }

	set0_cmpfvector(ret);
    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
		#pragma omp critical
		{
			add_cmpfvector(ret, ret, work_vec[thread_index]);
			free_cmpfvector(work_vec[thread_index]);
		}
	}

	return SUCCESS;
}


// fix!
extern void _bnc_mpc_mul_cd(mpc_t, mpc_t, double _Complex);

/* Multiply CDRSMatrix * CMPFVector */
int _bncomp_mul_cdrsmatrix_cmpfvec(CMPFVector ret, CDRSMatrix mat, CMPFVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	unsigned long prec;
	mpc_t tmp[BNCOMP_MAX_NUM_THREADS];
    int thread_index;
	double _Complex mat_ij[BNCOMP_MAX_NUM_THREADS];

	if((ret->dim < mat->re->col_dim) || (vec->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
    	mpc_init2(tmp[thread_index], prec);

        //total_index[thread_index] = 0;
    }

	//total_index = 0;
    #pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < mat->re->row_dim; i++)
	{
        thread_index = omp_get_thread_num();

        // calculate accmulated total_index
        total_index[thread_index] = 0;
        for(j = 0; j < i; j++)
            total_index[thread_index] += mat->re->real_nzero_col_dim[j];

		//get_mpfvector_i(ret, i) = 0.0;
		mpc_set_ui(get_cmpfvector_i(ret, i), 0UL, MPC_RNDNN);
		for(j = 0; j < mat->re->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpf_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			//mpf_mul(tmp[thread_index], (mpf_ptr)(mat->element[total_index[thread_index]]), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mat_ij[thread_index] = mat->re->element[total_index[thread_index]] + mat->im->element[total_index[thread_index]] * I;
			//_bnc_mpc_mul_cd(tmp[thread_index], get_cmpfvector_i(vec, i), mat_ij[thread_index]);
			_bnc_mpc_mul_cd(tmp[thread_index], get_cmpfvector_i(vec, mat->re->nzero_index[i][j]), mat_ij[thread_index]);

			mpc_add(get_cmpfvector_i(ret, i), get_cmpfvector_i(ret, i), tmp[thread_index], MPC_RNDNN);

    		total_index[thread_index]++;
		}
    	total_index[thread_index] += (mat->re->real_nzero_col_dim[i] - mat->re->nzero_col_dim[i]);		
	}

    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
    	mpc_clear(tmp[thread_index]);
    }

	return SUCCESS;
}

/* Multiply CDRSMatrix^T * MPFVector */
int _bncomp_mul_cdrsmatrixt_cmpfvec(CMPFVector ret, CDRSMatrix mat, CMPFVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	unsigned long prec;
	mpc_t tmp[BNCOMP_MAX_NUM_THREADS], vi[BNCOMP_MAX_NUM_THREADS];
	CMPFVector work_vec[BNCOMP_MAX_NUM_THREADS];
    int thread_index, num_threads;
	long int div_row, start_i, end_i;
	double _Complex mat_ij[BNCOMP_MAX_NUM_THREADS];

	if((ret->dim < mat->re->col_dim) || (vec->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//prec = ret->prec;
	//mpf_init2(tmp, prec);
	prec = ret->prec;
    #pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

    	mpc_init2(tmp[thread_index], prec);
    	mpc_init2(vi[thread_index], prec);
		work_vec[thread_index] = init2_cmpfvector(vec->dim, prec);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->re->row_dim / num_threads;
		if(mat->re->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->re->row_dim) end_i = mat->re->row_dim;

		for(i = start_i; i < end_i; i++)
		{
			// calculate accmulated total_index
			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->re->real_nzero_col_dim[j];

			//#pragma omp section
			//#pragma omp critical			
			mpc_set(vi[thread_index], get_cmpfvector_i(vec, i), MPC_RNDNN);
			for(j = 0; j < mat->re->nzero_col_dim[i]; j++)
			{

		//#ifndef USE_MPFR
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				mat_ij[thread_index] = mat->re->element[total_index[thread_index]] + mat->im->element[total_index[thread_index]] * I;
				_bnc_mpc_mul_cd(tmp[thread_index], vi[thread_index], mat_ij[thread_index]); // get_mpfvector_i(vec, i));

				//#pragma omp critical
				mpc_add((mpc_ptr)(work_vec[thread_index]->element[mat->re->nzero_index[i][j]]), (mpc_ptr)(work_vec[thread_index]->element[mat->re->nzero_index[i][j]]), tmp[thread_index], MPC_RNDNN);
				//mpf_add(ret_tmp[thread_index], ret_tmp[thread_index], tmp[thread_index]);
		//#else
		/*
				//mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j), tmp[thread_index], bnc_default_rounding_mode);
				//#pragma omp critical
				mpfr_fma((mpf_ptr)(ret->element[mat->nzero_index[i][j]]), (mpf_ptr)(mat->element[total_index[thread_index]]), get_mpfvector_i(vec, i), (mpf_ptr)(ret->element[mat->nzero_index[i][j]]), MPFR_RNDN);
		*/
		//#endif
				total_index[thread_index]++;
			}
			//set_mpfvector_i(ret, i, ret_tmp[thread_index]);	
    		total_index[thread_index] += (mat->re->real_nzero_col_dim[i] - mat->re->nzero_col_dim[i]);		
		}

    	mpc_clear(tmp[thread_index]);
		mpc_clear(vi[thread_index]);
    }

	set0_cmpfvector(ret);
    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
		#pragma omp critical
		{
			add_cmpfvector(ret, ret, work_vec[thread_index]);
			free_cmpfvector(work_vec[thread_index]);
		}
	}

	return SUCCESS;
}

/* Multiply conj(CDRSMatrix)^T * MPFVector */
int _bncomp_mul_cdrsmatrixs_cmpfvec(CMPFVector ret, CDRSMatrix mat, CMPFVector vec)
{
	long int i, j, total_index[BNCOMP_MAX_NUM_THREADS];
	unsigned long prec;
	mpc_t tmp[BNCOMP_MAX_NUM_THREADS], vi[BNCOMP_MAX_NUM_THREADS];
	CMPFVector work_vec[BNCOMP_MAX_NUM_THREADS];
    int thread_index, num_threads;
	long int div_row, start_i, end_i;
	double _Complex mat_ij_conj[BNCOMP_MAX_NUM_THREADS];

	if((ret->dim < mat->re->col_dim) || (vec->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//prec = ret->prec;
	//mpf_init2(tmp, prec);
	prec = ret->prec;
    #pragma omp parallel private(thread_index, num_threads, div_row, start_i, end_i, i, j)
    {
        thread_index = omp_get_thread_num();
		num_threads = omp_get_num_threads();

    	mpc_init2(tmp[thread_index], prec);
    	mpc_init2(vi[thread_index], prec);
		work_vec[thread_index] = init2_cmpfvector(vec->dim, prec);

        //total_index[thread_index] = 0;
 
		//for(i = 0; i < mat->row_dim; i++)
		div_row = mat->re->row_dim / num_threads;
		if(mat->re->row_dim % num_threads != 0) div_row++;
		start_i = thread_index * div_row;
		end_i = start_i + div_row;
		if(end_i > mat->re->row_dim) end_i = mat->re->row_dim;

		for(i = start_i; i < end_i; i++)
		{
			// calculate accmulated total_index
			total_index[thread_index] = 0;
			for(j = 0; j < i; j++)
				total_index[thread_index] += mat->re->real_nzero_col_dim[j];

			//#pragma omp section
			//#pragma omp critical			
			mpc_set(vi[thread_index], get_cmpfvector_i(vec, i), MPC_RNDNN);
			for(j = 0; j < mat->re->nzero_col_dim[i]; j++)
			{

		//#ifndef USE_MPFR
				//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
				//mat_ij[thread_index] = mat->re->element[total_index[thread_index]] + mat->im->element[total_index[thread_index]] * I;
				mat_ij_conj[thread_index] = mat->re->element[total_index[thread_index]] - mat->im->element[total_index[thread_index]] * I;

				_bnc_mpc_mul_cd(tmp[thread_index], vi[thread_index], mat_ij_conj[thread_index]); // get_mpfvector_i(vec, i));

				//#pragma omp critical
				mpc_add((mpc_ptr)(work_vec[thread_index]->element[mat->re->nzero_index[i][j]]), (mpc_ptr)(work_vec[thread_index]->element[mat->re->nzero_index[i][j]]), tmp[thread_index], MPC_RNDNN);
				//mpf_add(ret_tmp[thread_index], ret_tmp[thread_index], tmp[thread_index]);
		//#else
		/*
				//mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j), tmp[thread_index], bnc_default_rounding_mode);
				//#pragma omp critical
				mpfr_fma((mpf_ptr)(ret->element[mat->nzero_index[i][j]]), (mpf_ptr)(mat->element[total_index[thread_index]]), get_mpfvector_i(vec, i), (mpf_ptr)(ret->element[mat->nzero_index[i][j]]), MPFR_RNDN);
		*/
		//#endif
				total_index[thread_index]++;
			}
			//set_mpfvector_i(ret, i, ret_tmp[thread_index]);	
    		total_index[thread_index] += (mat->re->real_nzero_col_dim[i] - mat->re->nzero_col_dim[i]);		
		}

    	mpc_clear(tmp[thread_index]);
		mpc_clear(vi[thread_index]);
    }

	set0_cmpfvector(ret);
    #pragma omp parallel private(thread_index)
    {
        thread_index = omp_get_thread_num();
		#pragma omp critical
		{
			add_cmpfvector(ret, ret, work_vec[thread_index]);
			free_cmpfvector(work_vec[thread_index]);
		}
	}

	return SUCCESS;
}
#endif // USE_GMP
