/********************************************************************************/
/* benchmark_tools.c: Functions for benchmarking                                */
/* Copyright (c) 2023 Tomonori Kouya                                            */
/*                                                                              */
/* Version 0.1, 2023-03-03: append relerr3_cmpfmatrix                           */
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
#include "matmul_strassen.h"

#ifdef USE_OMP
	#include "bncomp.h"
#endif // USE_OMP

// relative errors for double precision matrix
void relerr3_dmatrix(double *max_relerr, double *min_relerr, double *norm_relerr, DMatrix mat, DMatrix mat_true, int kind_of_norm)
{
	long int i, j, row_dim, col_dim;
	double tmp_relerr, mat_true_norm;
	DMatrix tmp_diff_mat;

	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	tmp_diff_mat = init_dmatrix(row_dim, col_dim);

	// diff_mat := mat - mat_true
	sub_dmatrix(tmp_diff_mat, mat, mat_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			*norm_relerr = normi_dmatrix(tmp_diff_mat);
			mat_true_norm = normi_dmatrix(mat_true);
			break;
		case 1: // 1-norm
			*norm_relerr = norm1_dmatrix(tmp_diff_mat);
			mat_true_norm = norm1_dmatrix(mat_true);
			break;
		case 2: // Frobenius norm
		default:
			*norm_relerr = normf_dmatrix(tmp_diff_mat);
			mat_true_norm = normf_dmatrix(mat_true);
			break;
	}
	if(mat_true_norm != 0.0)
		*norm_relerr /= mat_true_norm;

	// relative errors at each elements
	*max_relerr = 0.0;
	*min_relerr = 0.0;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			tmp_relerr = fabs(get_dmatrix_ij(tmp_diff_mat, i, j));
			if(get_dmatrix_ij(mat_true, i, j) != 0.0)
			{
				tmp_relerr /= get_dmatrix_ij(mat_true, i, j);
				tmp_relerr = fabs(tmp_relerr);
			}

			if(*max_relerr < tmp_relerr)
				*max_relerr = tmp_relerr;
			if(*min_relerr > tmp_relerr)
				*min_relerr = tmp_relerr;
		}
	}

	free_dmatrix(tmp_diff_mat);

	return;
}

// relative errors for double precision matrix
void relerr3_dvector(double *max_relerr, double *min_relerr, double *norm_relerr, DVector vec, DVector vec_true, int kind_of_norm)
{
	long int i, j, dim;
	double tmp_relerr, vec_true_norm;
	DVector tmp_diff_vec;

	dim = vec->dim;

	tmp_diff_vec = init_dvector(dim);

	// diff_mat := mat - mat_true
	sub_dvector(tmp_diff_vec, vec, vec_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			*norm_relerr = normi_dvector(tmp_diff_vec);
			vec_true_norm = normi_dvector(vec_true);
			break;
		case 1: // 1-norm
			*norm_relerr = norm1_dvector(tmp_diff_vec);
			vec_true_norm = norm1_dvector(vec_true);
			break;
		default:
			*norm_relerr = norm2_dvector(tmp_diff_vec);
			vec_true_norm = norm2_dvector(vec_true);
			break;
	}
	if(vec_true_norm != 0.0)
		*norm_relerr /= vec_true_norm;

	// relative errors at each elements
	*max_relerr = 0.0;
	*min_relerr = 0.0;

	for(i = 0; i < dim; i++)
	{
		tmp_relerr = fabs(get_dvector_i(tmp_diff_vec, i));
		if(get_dvector_i(vec_true, i) != 0.0)
		{
			tmp_relerr /= get_dvector_i(vec_true, i);
			tmp_relerr = fabs(tmp_relerr);
		}

		if(*max_relerr < tmp_relerr)
			*max_relerr = tmp_relerr;
		if(*min_relerr > tmp_relerr)
			*min_relerr = tmp_relerr;
	}

	free_dvector(tmp_diff_vec);

	return;
}

#ifdef USE_GMP
// relative errors for double precision vector
void relerr3_dvector_mpfvec(double *max_relerr, double *min_relerr, double *norm_relerr, DVector vec, MPFVector vec_true, int kind_of_norm)
{
    unsigned long prec;
    mpf_t in_max_relerr, in_min_relerr, in_norm_relerr;
    MPFVector in_vec;

    // Initialize
    prec = vec_true->prec;
    mpf_init2(in_max_relerr, prec);
    mpf_init2(in_min_relerr, prec);
    mpf_init2(in_norm_relerr, prec);
    in_vec = init2_mpfvector(vec->dim, prec);

    // (mpf_t)in_vec := vec
    subst_mpfvector_dvec(in_vec, vec);
    relerr3_mpfvector(in_max_relerr, in_min_relerr, in_norm_relerr, in_vec, vec_true, kind_of_norm);

    *max_relerr  = mpf_get_d(in_max_relerr);
    *min_relerr  = mpf_get_d(in_min_relerr);
    *norm_relerr = mpf_get_d(in_norm_relerr);

    // Free
    mpf_clear(in_max_relerr);
    mpf_clear(in_min_relerr);
    mpf_clear(in_norm_relerr);
    free_mpfvector(in_vec);
}

// relative errors for double precision matrix
void relerr3_dmatrix_mpfmat(double *max_relerr, double *min_relerr, double *norm_relerr, DMatrix mat, MPFMatrix mat_true, int kind_of_norm)
{
    unsigned long prec;
    mpf_t in_max_relerr, in_min_relerr, in_norm_relerr;
    MPFMatrix in_mat;

    // Initialize
    prec = mat_true->prec;
    mpf_init2(in_max_relerr, prec);
    mpf_init2(in_min_relerr, prec);
    mpf_init2(in_norm_relerr, prec);
    in_mat = init2_mpfmatrix(mat->row_dim, mat->col_dim, prec);

    // (mpf_t)in_vec := vec
    subst_mpfmatrix_dmat(in_mat, mat);
    relerr3_mpfmatrix(in_max_relerr, in_min_relerr, in_norm_relerr, in_mat, mat_true, kind_of_norm);

    *max_relerr  = mpf_get_d(in_max_relerr);
    *min_relerr  = mpf_get_d(in_min_relerr);
    *norm_relerr = mpf_get_d(in_norm_relerr);

    // Free
    mpf_clear(in_max_relerr);
    mpf_clear(in_min_relerr);
    mpf_clear(in_norm_relerr);
    free_mpfmatrix(in_mat);
}

// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Frobenius norm
void relerr3_cdmatrix_cmpfmat(double *max_abs_relerr, double *min_abs_relerr, double *max_real_relerr, double *min_real_relerr, double *max_image_relerr, double *min_image_relerr, double *norm_relerr, CDMatrix mat, CMPFMatrix mat_true, int kind_of_norm)
{
    unsigned long prec;
    mpf_t in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr;
    CMPFMatrix in_mat;

    // Initialize
    prec = mat_true->prec;
    mpf_init2(in_max_abs_relerr, prec);
    mpf_init2(in_min_abs_relerr, prec);
    mpf_init2(in_max_real_relerr, prec);
    mpf_init2(in_min_real_relerr, prec);
    mpf_init2(in_max_image_relerr, prec);
    mpf_init2(in_min_image_relerr, prec);
    mpf_init2(in_norm_relerr, prec);
    in_mat = init2_cmpfmatrix(mat->row_dim, mat->col_dim, prec);

    // (mpf_t)in_vec := vec
    subst_cmpfmatrix_cdmat(in_mat, mat);
    relerr3_cmpfmatrix(in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr, in_mat, mat_true, kind_of_norm);

    *max_abs_relerr   = mpf_get_d(in_max_abs_relerr);
    *min_abs_relerr   = mpf_get_d(in_min_abs_relerr);
    *max_real_relerr  = mpf_get_d(in_max_real_relerr);
    *min_real_relerr  = mpf_get_d(in_min_real_relerr);
    *max_image_relerr = mpf_get_d(in_max_image_relerr);
    *min_image_relerr = mpf_get_d(in_min_image_relerr);
    *norm_relerr      = mpf_get_d(in_norm_relerr);

    // Free
    mpf_clear(in_max_abs_relerr);
    mpf_clear(in_min_abs_relerr);
    mpf_clear(in_max_real_relerr);
    mpf_clear(in_min_real_relerr);
    mpf_clear(in_max_image_relerr);
    mpf_clear(in_min_image_relerr);
    mpf_clear(in_norm_relerr);
    free_cmpfmatrix(in_mat);
}

// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Euclieian norm
void relerr3_cdvector_cmpfvec(double *max_abs_relerr, double *min_abs_relerr, double *max_real_relerr, double *min_real_relerr, double *max_image_relerr, double *min_image_relerr, double *norm_relerr, CDVector vec, CMPFVector vec_true, int kind_of_norm)
{
    unsigned long prec;
    mpf_t in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr;
    CMPFVector in_vec;

    // Initialize
    prec = vec_true->prec;
    mpf_init2(in_max_abs_relerr, prec);
    mpf_init2(in_min_abs_relerr, prec);
    mpf_init2(in_max_real_relerr, prec);
    mpf_init2(in_min_real_relerr, prec);
    mpf_init2(in_max_image_relerr, prec);
    mpf_init2(in_min_image_relerr, prec);
    mpf_init2(in_norm_relerr, prec);
    in_vec = init2_cmpfvector(vec->dim, prec);

    // (mpf_t)in_vec := vec
    subst_cmpfvector_cdvec(in_vec, vec);
    relerr3_cmpfvector(in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr, in_vec, vec_true, kind_of_norm);

    *max_abs_relerr   = mpf_get_d(in_max_abs_relerr);
    *min_abs_relerr   = mpf_get_d(in_min_abs_relerr);
    *max_real_relerr  = mpf_get_d(in_max_real_relerr);
    *min_real_relerr  = mpf_get_d(in_min_real_relerr);
    *max_image_relerr = mpf_get_d(in_max_image_relerr);
    *min_image_relerr = mpf_get_d(in_min_image_relerr);
    *norm_relerr      = mpf_get_d(in_norm_relerr);

    // Free
    mpf_clear(in_max_abs_relerr);
    mpf_clear(in_min_abs_relerr);
    mpf_clear(in_max_real_relerr);
    mpf_clear(in_min_real_relerr);
    mpf_clear(in_max_image_relerr);
    mpf_clear(in_min_image_relerr);
    mpf_clear(in_norm_relerr);
    free_cmpfvector(in_vec);
}


// relative errors
void relerr3_mpfvector(mpf_t max_relerr, mpf_t min_relerr, mpf_t norm_relerr, MPFVector vec, MPFVector vec_true, int kind_of_norm)
{
	unsigned long prec;
	long int i, j, dim;
	mpf_t tmp_relerr, vec_true_norm;
	MPFVector tmp_diff_vec;

	prec = vec->prec;
	dim = vec->dim;

	mpf_init2(tmp_relerr, prec);
	mpf_init2(vec_true_norm, prec);
	tmp_diff_vec = init2_mpfvector(dim, prec);

	// diff_vec := vec - vec_true
	sub_mpfvector(tmp_diff_vec, vec, vec_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_mpfvector(norm_relerr, tmp_diff_vec);
			normi_mpfvector(vec_true_norm, vec_true);
			break;
		case 1: // 1-norm
			norm1_mpfvector(norm_relerr, tmp_diff_vec);
			norm1_mpfvector(vec_true_norm, vec_true);
			break;
		default:
			norm2_mpfvector(norm_relerr, tmp_diff_vec);
			norm2_mpfvector(vec_true_norm, vec_true);
			break;
	}
	if(mpf_cmp_ui(vec_true_norm, 0UL) != 0)
		mpf_div(norm_relerr, norm_relerr, vec_true_norm);

	// relative errors at each elements
	mpf_set_ui(max_relerr, 0UL);
	mpf_set_ui(min_relerr, 0UL);

	for(i = 0; i < dim; i++)
	{
		mpf_abs(tmp_relerr, get_mpfvector_i(tmp_diff_vec, i));
		if(mpf_cmp_ui(get_mpfvector_i(vec_true, i), 0UL) != 0)
		{
			mpf_div(tmp_relerr, tmp_relerr, get_mpfvector_i(vec_true, i));
			mpf_abs(tmp_relerr, tmp_relerr);
		}

		if(mpf_cmp(max_relerr, tmp_relerr) < 0)
			mpf_set(max_relerr, tmp_relerr);
		if(mpf_cmp(min_relerr, tmp_relerr) > 0)
			mpf_set(min_relerr, tmp_relerr);
	}

	mpf_clear(tmp_relerr);
	mpf_clear(vec_true_norm);
	
	free_mpfvector(tmp_diff_vec);

	return;
}

// relative errors
void relerr3_mpfmatrix(mpf_t max_relerr, mpf_t min_relerr, mpf_t norm_relerr, MPFMatrix mat, MPFMatrix mat_true, int kind_of_norm)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim;
	mpf_t tmp_relerr, mat_true_norm;
	MPFMatrix tmp_diff_mat;

	prec = mat->prec;
	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	mpf_init2(tmp_relerr, prec);
	mpf_init2(mat_true_norm, prec);
	tmp_diff_mat = init2_mpfmatrix(row_dim, col_dim, prec);

	// diff_mat := mat - mat_true
	sub_mpfmatrix(tmp_diff_mat, mat, mat_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_mpfmatrix(norm_relerr, tmp_diff_mat);
			normi_mpfmatrix(mat_true_norm, mat_true);
			break;
		case 1: // 1-norm
			norm1_mpfmatrix(norm_relerr, tmp_diff_mat);
			norm1_mpfmatrix(mat_true_norm, mat_true);
			break;
		case 2: // Frobenius norm
		default:
			normf_mpfmatrix(norm_relerr, tmp_diff_mat);
			normf_mpfmatrix(mat_true_norm, mat_true);
			break;
	}
	if(mpf_cmp_ui(mat_true_norm, 0UL) != 0)
		mpf_div(norm_relerr, norm_relerr, mat_true_norm);

	// relative errors at each elements
	mpf_set_ui(max_relerr, 0UL);
	mpf_set_ui(min_relerr, 0UL);

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			mpf_abs(tmp_relerr, get_mpfmatrix_ij(tmp_diff_mat, i, j));
			if(mpf_cmp_ui(get_mpfmatrix_ij(mat_true, i, j), 0UL) != 0)
			{
				mpf_div(tmp_relerr, tmp_relerr, get_mpfmatrix_ij(mat_true, i, j));
				mpf_abs(tmp_relerr, tmp_relerr);
			}

			if(mpf_cmp(max_relerr, tmp_relerr) < 0)
				mpf_set(max_relerr, tmp_relerr);
			if(mpf_cmp(min_relerr, tmp_relerr) > 0)
				mpf_set(min_relerr, tmp_relerr);
		}
	}

	mpf_clear(tmp_relerr);
	mpf_clear(mat_true_norm);
	
	free_mpfmatrix(tmp_diff_mat);

	return;
}

// relative errors
void relerr3_cmpfmatrix_old(mpf_t max_relerr, mpf_t min_relerr, mpf_t norm_relerr, CMPFMatrix mat, CMPFMatrix mat_true, int kind_of_norm)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim;
	mpf_t tmp_relerr, mat_true_norm, tmp;
	CMPFMatrix tmp_diff_mat;

	prec = mat->prec;
	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	mpf_init2(tmp_relerr, prec);
	mpf_init2(mat_true_norm, prec);
	mpf_init2(tmp, prec);
	tmp_diff_mat = init2_cmpfmatrix(row_dim, col_dim, prec);

	// diff_mat := mat - mat_true
	sub_cmpfmatrix(tmp_diff_mat, mat, mat_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_cmpfmatrix(norm_relerr, tmp_diff_mat);
			normi_cmpfmatrix(mat_true_norm, mat_true);
			break;
		case 1: // 1-norm
			norm1_cmpfmatrix(norm_relerr, tmp_diff_mat);
			norm1_cmpfmatrix(mat_true_norm, mat_true);
			break;
		case 2: // Frobenius norm
		default:
			normf_cmpfmatrix(norm_relerr, tmp_diff_mat);
			normf_cmpfmatrix(mat_true_norm, mat_true);
			break;
	}
	if(mpf_cmp_ui(mat_true_norm, 0UL) != 0)
		mpf_div(norm_relerr, norm_relerr, mat_true_norm);

	// relative errors at each elements
	mpf_set_ui(max_relerr, 0UL);
	mpf_set_ui(min_relerr, 0UL);

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
#ifdef USE_MPFCMPLX
			//mpf_abs(tmp_relerr, get_mpfmatrix_ij(tmp_diff_mat, i, j));
			abs_mpfcmplx(tmp_relerr, get_cmpfmatrix_ij(tmp_diff_mat, i, j));
			if((mpf_cmp_ui(getp_real_mpfcmplx(get_cmpfmatrix_ij(mat_true, i, j)), 0UL) != 0) || (mpf_cmp_ui(getp_image_mpfcmplx(get_cmpfmatrix_ij(mat_true, i, j)), 0UL) != 0))
			{
				//mpf_div(tmp_relerr, tmp_relerr, get_mpfmatrix_ij(mat_true, i, j));
				abs_mpfcmplx(tmp, get_cmpfmatrix_ij(mat_true, i, j));
				mpf_div(tmp_relerr, tmp_relerr, tmp);
				//mpf_abs(tmp_relerr, tmp_relerr);
			}
#else // USE_MPFCMPLX
			mpc_abs(tmp_relerr, get_cmpfmatrix_ij(tmp_diff_mat, i, j), get_bnc_default_rounding_mode());
			//abs_mpfcmplx(tmp_relerr, get_cmpfmatrix_ij(tmp_diff_mat, i, j));
			if((mpf_cmp_ui(mpc_realref(get_cmpfmatrix_ij(mat_true, i, j)), 0UL) != 0) || (mpf_cmp_ui(mpc_imagref(get_cmpfmatrix_ij(mat_true, i, j)), 0UL) != 0))
			{
				//mpf_div(tmp_relerr, tmp_relerr, get_mpfmatrix_ij(mat_true, i, j));
				mpc_abs(tmp, get_cmpfmatrix_ij(mat_true, i, j), get_bnc_default_rounding_mode());
				mpf_div(tmp_relerr, tmp_relerr, tmp);
				//mpf_abs(tmp_relerr, tmp_relerr);
			}
#endif // USE_MPFCMPLX

			if(mpf_cmp(max_relerr, tmp_relerr) < 0)
				mpf_set(max_relerr, tmp_relerr);
			if(mpf_cmp(min_relerr, tmp_relerr) > 0)
				mpf_set(min_relerr, tmp_relerr);
		}
	}

	mpf_clear(tmp_relerr);
	mpf_clear(mat_true_norm);
	mpf_clear(tmp);
	
	free_cmpfmatrix(tmp_diff_mat);

	return;
}
// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Frobenius norm
void relerr3_cmpfmatrix(mpf_t max_abs_relerr, mpf_t min_abs_relerr, mpf_t max_real_relerr, mpf_t min_real_relerr, mpf_t max_image_relerr, mpf_t min_image_relerr, mpf_t norm_relerr, CMPFMatrix mat, CMPFMatrix mat_true, int kind_of_norm)
{
	unsigned long prec;
	long int i, j, k, row_dim, col_dim;
	mpf_t tmp_relerr[3], mat_true_norm, tmp;
	CMPFMatrix tmp_diff_mat;

	prec = mat->prec;
	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	mpf_init2(tmp_relerr[0], prec);
	mpf_init2(tmp_relerr[1], prec);
	mpf_init2(tmp_relerr[2], prec);
	mpf_init2(mat_true_norm, prec);
	mpf_init2(tmp, prec);
	tmp_diff_mat = init2_cmpfmatrix(row_dim, col_dim, prec);

	// diff_mat := mat - mat_true
	sub_cmpfmatrix(tmp_diff_mat, mat, mat_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinite norm
			normi_cmpfmatrix(norm_relerr, tmp_diff_mat);
			normi_cmpfmatrix(mat_true_norm, mat_true);
			break;
		case 1: // 1-norm
			norm1_cmpfmatrix(norm_relerr, tmp_diff_mat);
			norm1_cmpfmatrix(mat_true_norm, mat_true);
			break;
		case 2: // Frobenius norm
		default:
			normf_cmpfmatrix(norm_relerr, tmp_diff_mat);
			normf_cmpfmatrix(mat_true_norm, mat_true);
			break;
	}
	if(mpf_cmp_ui(mat_true_norm, 0UL) != 0)
		mpf_div(norm_relerr, norm_relerr, mat_true_norm);

	// relative errors at each elements
	mpf_set_ui(max_abs_relerr, 0UL);
	mpf_set_ui(min_abs_relerr, 0UL);
	mpf_set_ui(max_real_relerr, 0UL);
	mpf_set_ui(min_real_relerr, 0UL);
	mpf_set_ui(max_image_relerr, 0UL);
	mpf_set_ui(min_image_relerr, 0UL);

// relative error of absolute values
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
#ifdef USE_MPFCMPLX
			// Absolute value
			//mpf_abs(tmp_relerr, get_mpfmatrix_ij(tmp_diff_mat, i, j));
			abs_mpfcmplx(tmp_relerr[0], get_cmpfmatrix_ij(tmp_diff_mat, i, j));
			if((mpf_cmp_ui(getp_real_mpfcmplx(get_cmpfmatrix_ij(mat_true, i, j)), 0UL) != 0) || (mpf_cmp_ui(getp_image_mpfcmplx(get_cmpfmatrix_ij(mat_true, i, j)), 0UL) != 0))
			{
				//mpf_div(tmp_relerr, tmp_relerr, get_mpfmatrix_ij(mat_true, i, j));
				abs_mpfcmplx(tmp, get_cmpfmatrix_ij(mat_true, i, j));
				mpf_div(tmp_relerr[0], tmp_relerr[0], tmp);
				//mpf_abs(tmp_relerr, tmp_relerr);
			}
			// real part 
			tmp_relerr[1] = getp_real_mpfcmplx(get_cmpfmatrix_ij(tmp_diff_mat, i, j)));
			tmp = getp_real_mpfcmplx(get_cmpfmatrix_ij(mat_true, i, j));
			if(mpf_cmp_ui(tmp, 0UL) != 0)
				mpf_div(tmp_relerr[1], tmp_relerr[1], tmp);

			mpf_abs(tmp_relerr[1], tmp_relerr[1]);

			// imaginary part 
			tmp_relerr[2] = getp_image_mpfcmplx(get_cmpfmatrix_ij(tmp_diff_mat, i, j)));
			tmp = getp_image_mpfcmplx(get_cmpfmatrix_ij(mat_true, i, j));
			if(mpf_cmp_ui(tmp, 0UL) != 0)
				mpf_div(tmp_relerr[2], tmp_relerr[2], tmp);

			mpf_abs(tmp_relerr[2], tmp_relerr[2]);


#else // USE_MPFCMPLX
			// Absolute value
			mpc_abs(tmp_relerr[0], get_cmpfmatrix_ij(tmp_diff_mat, i, j), get_bnc_default_rounding_mode());
			//abs_mpfcmplx(tmp_relerr, get_cmpfmatrix_ij(tmp_diff_mat, i, j));
			if((mpf_cmp_ui(mpc_realref(get_cmpfmatrix_ij(mat_true, i, j)), 0UL) != 0) || (mpf_cmp_ui(mpc_imagref(get_cmpfmatrix_ij(mat_true, i, j)), 0UL) != 0))
			{
				//mpf_div(tmp_relerr, tmp_relerr, get_mpfmatrix_ij(mat_true, i, j));
				mpc_abs(tmp, get_cmpfmatrix_ij(mat_true, i, j), get_bnc_default_rounding_mode());
				mpf_div(tmp_relerr[0], tmp_relerr[0], tmp);
				//mpf_abs(tmp_relerr, tmp_relerr);
			}

			// Real part
			mpf_set(tmp_relerr[1], mpc_realref(get_cmpfmatrix_ij(tmp_diff_mat, i, j)));
			mpf_set(tmp, mpc_realref(get_cmpfmatrix_ij(mat_true, i, j)));
			if(mpf_cmp_ui(tmp, 0UL) != 0)
				mpf_div(tmp_relerr[1], tmp_relerr[1], tmp);
			mpf_abs(tmp_relerr[1], tmp_relerr[1]);

			// Imaginary part
			mpf_set(tmp_relerr[2], mpc_imagref(get_cmpfmatrix_ij(tmp_diff_mat, i, j)));
			mpf_set(tmp, mpc_imagref(get_cmpfmatrix_ij(mat_true, i, j)));
			if(mpf_cmp_ui(tmp, 0UL) != 0)
				mpf_div(tmp_relerr[2], tmp_relerr[2], tmp);
			mpf_abs(tmp_relerr[2], tmp_relerr[2]);

#endif // USE_MPFCMPLX

			// Absolute value
			if(mpf_cmp(max_abs_relerr, tmp_relerr[0]) < 0)
				mpf_set(max_abs_relerr, tmp_relerr[0]);
			if(mpf_cmp(min_abs_relerr, tmp_relerr[0]) > 0)
				mpf_set(min_abs_relerr, tmp_relerr[0]);

			// Real part
			if(mpf_cmp(max_real_relerr, tmp_relerr[1]) < 0)
				mpf_set(max_real_relerr, tmp_relerr[1]);
			if(mpf_cmp(min_real_relerr, tmp_relerr[1]) > 0)
				mpf_set(min_real_relerr, tmp_relerr[1]);

			// Imaginary value
			if(mpf_cmp(max_image_relerr, tmp_relerr[2]) < 0)
				mpf_set(max_image_relerr, tmp_relerr[2]);
			if(mpf_cmp(min_image_relerr, tmp_relerr[2]) > 0)
				mpf_set(min_image_relerr, tmp_relerr[2]);

		}
	}

	mpf_clear(tmp_relerr[0]);
	mpf_clear(tmp_relerr[1]);
	mpf_clear(tmp_relerr[2]);
	mpf_clear(mat_true_norm);
	mpf_clear(tmp);
	
	free_cmpfmatrix(tmp_diff_mat);

	return;
}
// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Frobenius norm
void relerr3_cmpfvector(mpf_t max_abs_relerr, mpf_t min_abs_relerr, mpf_t max_real_relerr, mpf_t min_real_relerr, mpf_t max_image_relerr, mpf_t min_image_relerr, mpf_t norm_relerr, CMPFVector vec, CMPFVector vec_true, int kind_of_norm)
{
	unsigned long prec;
	long int i, j, dim;
	mpf_t tmp_relerr[3], true_norm, tmp;
	CMPFVector tmp_diff_vec;

	prec = vec->prec;
	dim = vec->dim;

	mpf_init2(tmp_relerr[0], prec);
	mpf_init2(tmp_relerr[1], prec);
	mpf_init2(tmp_relerr[2], prec);
	mpf_init2(true_norm, prec);
	mpf_init2(tmp, prec);
	tmp_diff_vec = init2_cmpfvector(dim, prec);

	// diff_mat := mat - mat_true
	sub_cmpfvector(tmp_diff_vec, vec, vec_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinite norm
			normi_cmpfvector(norm_relerr, tmp_diff_vec);
			normi_cmpfvector(true_norm, vec_true);
			break;
		case 1: // 1-norm
			norm1_cmpfvector(norm_relerr, tmp_diff_vec);
			norm1_cmpfvector(true_norm, vec_true);
			break;
		case 2: // Euclidian norm
		default:
			norm2_cmpfvector(norm_relerr, tmp_diff_vec);
			norm2_cmpfvector(true_norm, vec_true);
			break;
	}
	if(mpf_cmp_ui(true_norm, 0UL) != 0)
		mpf_div(norm_relerr, norm_relerr, true_norm);

	// relative errors at each elements
	mpf_set_ui(max_abs_relerr, 0UL);
	mpf_set_ui(min_abs_relerr, 0UL);
	mpf_set_ui(max_real_relerr, 0UL);
	mpf_set_ui(min_real_relerr, 0UL);
	mpf_set_ui(max_image_relerr, 0UL);
	mpf_set_ui(min_image_relerr, 0UL);

// relative error of absolute values
	for(i = 0; i < dim; i++)
	{
#ifdef USE_MPFCMPLX
		// Absolute value
		abs_mpfcmplx(tmp_relerr[0], get_cmpfvector_i(tmp_diff_vec, i));
		if((mpf_cmp_ui(getp_real_mpfcmplx(get_cmpfvector_i(vec_true, i)), 0UL) != 0) || (mpf_cmp_ui(getp_image_mpfcmplx(get_cmpfvector_i(vec_true, i)), 0UL) != 0))
		{
			//mpf_div(tmp_relerr, tmp_relerr, get_mpfmatrix_ij(mat_true, i, j));
			abs_mpfcmplx(tmp, get_cmpfvector_i(vec_true, i));
			mpf_div(tmp_relerr[0], tmp_relerr[0], tmp);
			//mpf_abs(tmp_relerr, tmp_relerr);
		}
		// real part 
		tmp_relerr = getp_real_mpfcmplx(get_cmpfvector_i(tmp_diff_vec, i)));
		tmp = getp_real_mpfcmplx(get_cmpfvector_i(vec_true, i));
		if(mpf_cmp_ui(tmp, 0UL) != 0)
			mpf_div(tmp_relerr[1], tmp_relerr[1], tmp);
		mpf_abs(tmp_relerr[1], tmp_relerr[1]);
		// imaginary part 
		tmp_relerr = getp_image_mpfcmplx(get_cmpfvector_i(tmp_diff_vec, i));
		tmp = getp_image_mpfcmplx(get_cmpfvector_i(vec_true, i));
		if(mpf_cmp_ui(tmp, 0UL) != 0)
			mpf_div(tmp_relerr[2], tmp_relerr[2], tmp);
		mpf_abs(tmp_relerr[2], tmp_relerr[2]);

#else // USE_MPFCMPLX
		// Absolute value
		mpc_abs(tmp_relerr[0], get_cmpfvector_i(tmp_diff_vec, i), get_bnc_default_rounding_mode());
		//abs_mpfcmplx(tmp_relerr, get_cmpfvector_ij(tmp_diff_mat, i, j));
		if((mpf_cmp_ui(mpc_realref(get_cmpfvector_i(vec_true, i)), 0UL) != 0) || (mpf_cmp_ui(mpc_imagref(get_cmpfvector_i(vec_true, i)), 0UL) != 0))
		{
			//mpf_div(tmp_relerr, tmp_relerr, get_mpfmatrix_ij(mat_true, i, j));
			mpc_abs(tmp, get_cmpfvector_i(vec_true, i), get_bnc_default_rounding_mode());
			mpf_div(tmp_relerr[0], tmp_relerr[0], tmp);
			//mpf_abs(tmp_relerr, tmp_relerr);
		}
		// Real part
		mpf_set(tmp_relerr[1], mpc_realref(get_cmpfvector_i(tmp_diff_vec, i)));
		mpf_set(tmp, mpc_realref(get_cmpfvector_i(vec_true, i)));
		if(mpf_cmp_ui(tmp, 0UL) != 0)
			mpf_div(tmp_relerr[1], tmp_relerr[1], tmp);
		mpf_abs(tmp_relerr[1], tmp_relerr[1]);
		// Imaginary part
		mpf_set(tmp_relerr[2], mpc_imagref(get_cmpfvector_i(tmp_diff_vec, i)));
		mpf_set(tmp, mpc_imagref(get_cmpfvector_i(vec_true, i)));
		if(mpf_cmp_ui(tmp, 0UL) != 0)
			mpf_div(tmp_relerr[2], tmp_relerr[2], tmp);
		mpf_abs(tmp_relerr[2], tmp_relerr[2]);
#endif // USE_MPFCMPLX
		// Absolute value
		if(mpf_cmp(max_abs_relerr, tmp_relerr[0]) < 0)
			mpf_set(max_abs_relerr, tmp_relerr[0]);
		if(mpf_cmp(min_abs_relerr, tmp_relerr[0]) > 0)
			mpf_set(min_abs_relerr, tmp_relerr[0]);
		// Real part
		if(mpf_cmp(max_real_relerr, tmp_relerr[1]) < 0)
			mpf_set(max_real_relerr, tmp_relerr[1]);
		if(mpf_cmp(min_real_relerr, tmp_relerr[1]) > 0)
			mpf_set(min_real_relerr, tmp_relerr[1]);
		// Imaginary value
		if(mpf_cmp(max_image_relerr, tmp_relerr[2]) < 0)
			mpf_set(max_image_relerr, tmp_relerr[2]);
		if(mpf_cmp(min_image_relerr, tmp_relerr[2]) > 0)
			mpf_set(min_image_relerr, tmp_relerr[2]);
	}

	mpf_clear(tmp_relerr[0]);
	mpf_clear(tmp_relerr[1]);
	mpf_clear(tmp_relerr[2]);
	mpf_clear(true_norm);
	mpf_clear(tmp);
	
	free_cmpfvector(tmp_diff_vec);

	return;
}
#endif // USE_GMP
