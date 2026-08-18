//#include "bnc.h"
#include "matmul_strassen.h"

#ifdef USE_OMP
	#include "bncomp.h"
#endif // USE_OMP

// DD and QD
//#include "ddlinear.h"

// ANSI C
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifdef USE_DDLINEAR

// relative errors of vector
void relerr3_ddvector(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDVector vec, DDVector vec_true, int kind_of_norm)
{
	long int i, j, dim;
	double tmp_relerr[DDSIZE], vec_true_norm[DDSIZE];
	DDVector tmp_diff_vec;

	dim = vec->dim;

	tmp_diff_vec = init_ddvector(dim);

	// diff_mat := mat - mat_true
	sub_ddvector(tmp_diff_vec, vec, vec_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_ddvector(norm_relerr, tmp_diff_vec);
			normi_ddvector(vec_true_norm, vec_true);
			break;
		case 1: // 1-norm
			norm1_ddvector(norm_relerr, tmp_diff_vec);
			norm1_ddvector(vec_true_norm, vec_true);
			break;
		default:
			norm2_ddvector(norm_relerr, tmp_diff_vec);
			norm2_ddvector(vec_true_norm, vec_true);
			break;
	}
	if(rdd_cmp_ui(vec_true_norm, 0UL) != 0)
		rdd_div(norm_relerr, norm_relerr, vec_true_norm);

	// relative errors at each elements
	rdd_set_ui(max_relerr, 0UL);
	rdd_set_ui(min_relerr, 0UL);

	for(i = 0; i < dim; i++)
	{
		rdd_abs(tmp_relerr, get_ddvector_i(tmp_diff_vec, i));
		if(rdd_cmp_ui(get_ddvector_i(vec_true, i), 0UL) != 0)
		{
			rdd_div(tmp_relerr, tmp_relerr, get_ddvector_i(vec_true, i));
			rdd_abs(tmp_relerr, tmp_relerr);
		}

		if(rdd_cmp(max_relerr, tmp_relerr) < 0)
			rdd_set(max_relerr, tmp_relerr);
		if(rdd_cmp(min_relerr, tmp_relerr) > 0)
			rdd_set(min_relerr, tmp_relerr);
	}

	free_ddvector(tmp_diff_vec);

	return;
}

// relative errors of complex vector
void relerr3_cddvector(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double max_real_relerr[DDSIZE], double min_real_relerr[DDSIZE], double max_image_relerr[DDSIZE], double min_image_relerr[DDSIZE], double norm_relerr[DDSIZE], CDDVector vec, CDDVector vec_true, int kind_of_norm)
{
	long int i, j, dim;
	ddfloat tmp_relerr[7], tmp_abs;
	double vec_true_norm[DDSIZE];
	cddfloat ctmp;
	CDDVector tmp_diff_vec;

	dim = vec->re->dim;

	tmp_diff_vec = init_cddvector(dim);

	// diff_vec := vec - vec_true
	sub_cddvector(tmp_diff_vec, vec, vec_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_cddvector(norm_relerr, tmp_diff_vec);
			normi_cddvector(vec_true_norm, vec_true);
			break;
		case 1: // 1-norm
			norm1_cddvector(norm_relerr, tmp_diff_vec);
			norm1_cddvector(vec_true_norm, vec_true);
			break;
		case 2: // 2-norm
		default:
			norm2_cddvector(norm_relerr, tmp_diff_vec);
			norm2_cddvector(vec_true_norm, vec_true);
			break;
	}
	if(rdd_cmp_ui(vec_true_norm, 0UL) != 0)
		rdd_div(norm_relerr, norm_relerr, vec_true_norm);

	// relative errors at each elements
	rdd_set_ui(max_relerr, 0UL);
	rdd_set_ui(min_relerr, 0UL);
	rdd_set_ui(max_real_relerr, 0UL);
	rdd_set_ui(min_real_relerr, 0UL);
	rdd_set_ui(max_image_relerr, 0UL);
	rdd_set_ui(min_image_relerr, 0UL);
	for(i = 0; i < dim; i++)
	{
		// Absolute value
		//rcdd_abs(&tmp_relerr[0], get_cddvector_i(tmp_diff_vec, i));
		//rcdd_abs(&tmp_abs, get_cddvector_i(vec_true, i));
		rdd_set(ctmp.val_re, get_ddvector_i(tmp_diff_vec->re, i));
		rdd_set(ctmp.val_im, get_ddvector_i(tmp_diff_vec->im, i));
		rcdd_abs(&tmp_relerr[0], &ctmp);
		rdd_set(ctmp.val_re, get_ddvector_i(vec_true->re, i));
		rdd_set(ctmp.val_im, get_ddvector_i(vec_true->im, i));
		rcdd_abs(&tmp_abs, &ctmp);

		if(rdd_cmp_ui(tmp_abs.val, 0UL) != 0)
		{
			rdd_div(tmp_relerr[0].val, tmp_relerr[0].val, tmp_abs.val);
			//rdd_abs(tmp_relerr.val, tmp_relerr.val);
		}

		// real part 
		rdd_abs(tmp_relerr[1].val, get_ddvector_i(tmp_diff_vec->re, i));
		rdd_abs(tmp_abs.val, get_ddvector_i(vec_true->re, i));
		if(rdd_cmp_ui(tmp_abs.val, 0UL) != 0)
			rdd_div(tmp_relerr[1].val, tmp_relerr[1].val, tmp_abs.val);

		//rdd_abs(tmp_relerr[1].val, tmp_relerr[1].val);

		// imaginary part 
		rdd_abs(tmp_relerr[2].val, get_ddvector_i(tmp_diff_vec->im, i));
		rdd_abs(tmp_abs.val, get_ddvector_i(vec_true->im, i));
		if(rdd_cmp_ui(tmp_abs.val, 0UL) != 0)
			rdd_div(tmp_relerr[2].val, tmp_relerr[2].val, tmp_abs.val);

		//rdd_abs(tmp_relerr[2].val, tmp_relerr21].val);

		if(rdd_cmp(max_relerr, tmp_relerr[0].val) < 0) rdd_set(max_relerr, tmp_relerr[0].val);
		if(rdd_cmp(min_relerr, tmp_relerr[0].val) > 0) rdd_set(min_relerr, tmp_relerr[0].val);
		if(rdd_cmp(max_real_relerr, tmp_relerr[1].val) < 0) rdd_set(max_real_relerr, tmp_relerr[1].val);
		if(rdd_cmp(min_real_relerr, tmp_relerr[1].val) > 0) rdd_set(min_real_relerr, tmp_relerr[1].val);
		if(rdd_cmp(max_image_relerr, tmp_relerr[2].val) < 0) rdd_set(max_image_relerr, tmp_relerr[2].val);
		if(rdd_cmp(min_image_relerr, tmp_relerr[2].val) > 0) rdd_set(min_image_relerr, tmp_relerr[2].val);
	}

	free_cddvector(tmp_diff_vec);

	return;
}


// relative errors
void relerr3_ddmatrix(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDMatrix mat, DDMatrix mat_true, int kind_of_norm)
{
	long int i, j, row_dim, col_dim;
	double tmp_relerr[DDSIZE], mat_true_norm[DDSIZE];
	DDMatrix tmp_diff_mat;

	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	tmp_diff_mat = init_ddmatrix(row_dim, col_dim);

	// diff_mat := mat - mat_true
	sub_ddmatrix(tmp_diff_mat, mat, mat_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_ddmatrix(norm_relerr, tmp_diff_mat);
			normi_ddmatrix(mat_true_norm, mat_true);
			break;
		case 1: // 1-norm
			norm1_ddmatrix(norm_relerr, tmp_diff_mat);
			norm1_ddmatrix(mat_true_norm, mat_true);
			break;
		case 2: // Frobenius norm
		default:
			normf_ddmatrix(norm_relerr, tmp_diff_mat);
			normf_ddmatrix(mat_true_norm, mat_true);
			break;
	}
	if(rdd_cmp_ui(mat_true_norm, 0UL) != 0)
		rdd_div(norm_relerr, norm_relerr, mat_true_norm);

	// relative errors at each elements
	rdd_set_ui(max_relerr, 0UL);
	rdd_set_ui(min_relerr, 0UL);

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rdd_abs(tmp_relerr, get_ddmatrix_ij(tmp_diff_mat, i, j));
			if(rdd_cmp_ui(get_ddmatrix_ij(mat_true, i, j), 0UL) != 0)
			{
				rdd_div(tmp_relerr, tmp_relerr, get_ddmatrix_ij(mat_true, i, j));
				rdd_abs(tmp_relerr, tmp_relerr);
			}

			if(rdd_cmp(max_relerr, tmp_relerr) < 0)
				rdd_set(max_relerr, tmp_relerr);
			if(rdd_cmp(min_relerr, tmp_relerr) > 0)
				rdd_set(min_relerr, tmp_relerr);
		}
	}

	free_ddmatrix(tmp_diff_mat);

	return;
}

// relative errors of complex matrix
void relerr3_cddmatrix(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double max_real_relerr[DDSIZE], double min_real_relerr[DDSIZE], double max_image_relerr[DDSIZE], double min_image_relerr[DDSIZE], double norm_relerr[DDSIZE], CDDMatrix mat, CDDMatrix mat_true, int kind_of_norm)
{
	long int i, j, row_dim, col_dim;
	ddfloat tmp_relerr[3], tmp_abs;
	double mat_true_norm[DDSIZE];
	cddfloat ctmp;
	CDDMatrix tmp_diff_mat;

	row_dim = mat->re->row_dim;
	col_dim = mat->re->col_dim;

	tmp_diff_mat = init_cddmatrix(row_dim, col_dim);

	// diff_mat := mat - mat_true
	sub_cddmatrix(tmp_diff_mat, mat, mat_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_cddmatrix(norm_relerr, tmp_diff_mat);
			normi_cddmatrix(mat_true_norm, mat_true);
			break;
		case 1: // 1-norm
			norm1_cddmatrix(norm_relerr, tmp_diff_mat);
			norm1_cddmatrix(mat_true_norm, mat_true);
			break;
		case 2: // Frobenius norm
		default:
			normf_cddmatrix(norm_relerr, tmp_diff_mat);
			normf_cddmatrix(mat_true_norm, mat_true);
			break;
	}
	if(rdd_cmp_ui(mat_true_norm, 0UL) != 0)
		rdd_div(norm_relerr, norm_relerr, mat_true_norm);

	// relative errors at each elements
	rdd_set_ui(max_relerr, 0UL);
	rdd_set_ui(min_relerr, 0UL);
	rdd_set_ui(max_real_relerr, 0UL);
	rdd_set_ui(min_real_relerr, 0UL);
	rdd_set_ui(max_image_relerr, 0UL);
	rdd_set_ui(min_image_relerr, 0UL);

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			// Absolute value
			//rcdd_abs(&tmp_relerr[0], get_cddmatrix_ij(tmp_diff_mat, i, j));
			rdd_set(ctmp.val_re, get_ddmatrix_ij(tmp_diff_mat->re, i, j));
			rdd_set(ctmp.val_im, get_ddmatrix_ij(tmp_diff_mat->im, i, j));
			rcdd_abs(&tmp_relerr[0], &ctmp);
			//rcdd_abs(&tmp_abs, get_cddmatrix_ij(mat_true, i, j));
			rdd_set(ctmp.val_re, get_ddmatrix_ij(mat_true->re, i, j));
			rdd_set(ctmp.val_im, get_ddmatrix_ij(mat_true->im, i, j));
			rcdd_abs(&tmp_abs, &ctmp);

			if(rdd_cmp_ui(tmp_abs.val, 0UL) != 0)
			{
				rdd_div(tmp_relerr[0].val, tmp_relerr[0].val, tmp_abs.val); //get_cddmatrix_ij(mat_true, i, j));
				//rdd_abs(tmp_relerr.val, tmp_relerr.val);
			}

			// real part 
			rdd_abs(tmp_relerr[1].val, get_ddmatrix_ij(tmp_diff_mat->re, i, j));
			rdd_abs(tmp_abs.val, get_ddmatrix_ij(mat_true->re, i, j));
			if(rdd_cmp_ui(tmp_abs.val, 0UL) != 0)
				rdd_div(tmp_relerr[1].val, tmp_relerr[1].val, tmp_abs.val);

			//rdd_abs(tmp_relerr[1].val, tmp_relerr[1].val);

			// imaginary part 
			rdd_abs(tmp_relerr[2].val, get_ddmatrix_ij(tmp_diff_mat->im, i, j));
			rdd_abs(tmp_abs.val, get_ddmatrix_ij(mat_true->im, i, j));
			if(rdd_cmp_ui(tmp_abs.val, 0UL) != 0)
				rdd_div(tmp_relerr[2].val, tmp_relerr[2].val, tmp_abs.val);

			//rdd_abs(tmp_relerr[2].val, tmp_relerr21].val);

			if(rdd_cmp(max_relerr, tmp_relerr[0].val) < 0) rdd_set(max_relerr, tmp_relerr[0].val);
			if(rdd_cmp(min_relerr, tmp_relerr[0].val) > 0) rdd_set(min_relerr, tmp_relerr[0].val);
			if(rdd_cmp(max_real_relerr, tmp_relerr[1].val) < 0) rdd_set(max_real_relerr, tmp_relerr[1].val);
			if(rdd_cmp(min_real_relerr, tmp_relerr[1].val) > 0) rdd_set(min_real_relerr, tmp_relerr[1].val);
			if(rdd_cmp(max_image_relerr, tmp_relerr[2].val) < 0) rdd_set(max_image_relerr, tmp_relerr[2].val);
			if(rdd_cmp(min_image_relerr, tmp_relerr[2].val) > 0) rdd_set(min_image_relerr, tmp_relerr[2].val);

		}
	}

	free_cddmatrix(tmp_diff_mat);

	return;
}

#ifdef USE_GMP
// relative errors for double precision vector
void relerr3_ddvector_mpfvec(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDVector vec, MPFVector vec_true, int kind_of_norm)
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
    subst_mpfvector_ddvec(in_vec, vec);
    relerr3_mpfvector(in_max_relerr, in_min_relerr, in_norm_relerr, in_vec, vec_true, kind_of_norm);

    mpf_get_dd(max_relerr, in_max_relerr);
    mpf_get_dd(min_relerr, in_min_relerr);
    mpf_get_dd(norm_relerr, in_norm_relerr);

    // Free
    mpf_clear(in_max_relerr);
    mpf_clear(in_min_relerr);
    mpf_clear(in_norm_relerr);
    free_mpfvector(in_vec);
}

// relative errors for double precision matrix
void relerr3_ddmatrix_mpfmat(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDMatrix mat, MPFMatrix mat_true, int kind_of_norm)
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
    subst_mpfmatrix_ddmat(in_mat, mat);
    relerr3_mpfmatrix(in_max_relerr, in_min_relerr, in_norm_relerr, in_mat, mat_true, kind_of_norm);

    mpf_get_dd(max_relerr, in_max_relerr);
    mpf_get_dd(min_relerr, in_min_relerr);
    mpf_get_dd(norm_relerr, in_norm_relerr);

    // Free
    mpf_clear(in_max_relerr);
    mpf_clear(in_min_relerr);
    mpf_clear(in_norm_relerr);
    free_mpfmatrix(in_mat);
}

// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Frobenius norm
void relerr3_cddmatrix_cmpfmat(double max_abs_relerr[DDSIZE], double min_abs_relerr[DDSIZE], double max_real_relerr[DDSIZE], double min_real_relerr[DDSIZE], double max_image_relerr[DDSIZE], double min_image_relerr[DDSIZE], double norm_relerr[DDSIZE], CDDMatrix mat, CMPFMatrix mat_true, int kind_of_norm)
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
    in_mat = init2_cmpfmatrix(mat->re->row_dim, mat->re->col_dim, prec);

    // (mpf_t)in_vec := vec
    subst_cmpfmatrix_cddmat(in_mat, mat);
    relerr3_cmpfmatrix(in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr, in_mat, mat_true, kind_of_norm);

    mpf_get_dd(max_abs_relerr, in_max_abs_relerr);
    mpf_get_dd(min_abs_relerr, in_min_abs_relerr);
    mpf_get_dd(max_real_relerr, in_max_real_relerr);
    mpf_get_dd(min_real_relerr, in_min_real_relerr);
    mpf_get_dd(max_image_relerr, in_max_image_relerr);
    mpf_get_dd(min_image_relerr, in_min_image_relerr);
    mpf_get_dd(norm_relerr, in_norm_relerr);

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
void relerr3_cddvector_cmpfvec(double max_abs_relerr[DDSIZE], double min_abs_relerr[DDSIZE], double max_real_relerr[DDSIZE], double min_real_relerr[DDSIZE], double max_image_relerr[DDSIZE], double min_image_relerr[DDSIZE], double norm_relerr[DDSIZE], CDDVector vec, CMPFVector vec_true, int kind_of_norm)
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
    in_vec = init2_cmpfvector(vec->re->dim, prec);

    // (mpf_t)in_vec := vec
    subst_cmpfvector_cddvec(in_vec, vec);
    relerr3_cmpfvector(in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr, in_vec, vec_true, kind_of_norm);

    mpf_get_dd(max_abs_relerr, in_max_abs_relerr);
    mpf_get_dd(min_abs_relerr, in_min_abs_relerr);
    mpf_get_dd(max_real_relerr, in_max_real_relerr);
    mpf_get_dd(min_real_relerr, in_min_real_relerr);
    mpf_get_dd(max_image_relerr, in_max_image_relerr);
    mpf_get_dd(min_image_relerr, in_min_image_relerr);
    mpf_get_dd(norm_relerr, in_norm_relerr);

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

#endif // USE_GMP
#endif // USE_DDLINEAR

#ifdef USE_TDLINEAR

// relative errors of vector
void relerr3_tdvector(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDVector vec, TDVector vec_true, int kind_of_norm)
{
	long int i, j, dim;
	double tmp_relerr[TDSIZE], vec_true_norm[TDSIZE];
	TDVector tmp_diff_vec;

	dim = vec->dim;

	tmp_diff_vec= init_tdvector(dim);

	// diff_mat := mat - mat_true
	sub_tdvector(tmp_diff_vec, vec, vec_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_tdvector(norm_relerr, tmp_diff_vec);
			normi_tdvector(vec_true_norm, vec_true);
			break;
		case 1: // 1-norm
			norm1_tdvector(norm_relerr, tmp_diff_vec);
			norm1_tdvector(vec_true_norm, vec_true);
			break;
		default:
			norm2_tdvector(norm_relerr, tmp_diff_vec);
			norm2_tdvector(vec_true_norm, vec_true);
			break;
	}
	if(rtd_cmp_ui(vec_true_norm, 0UL) != 0)
		rtd_div(norm_relerr, norm_relerr, vec_true_norm);

	// relative errors at each elements
	rtd_set_ui(max_relerr, 0UL);
	rtd_set_ui(min_relerr, 0UL);

	for(i = 0; i < dim; i++)
	{
		rtd_abs(tmp_relerr, get_tdvector_i(tmp_diff_vec, i));
		if(rtd_cmp_ui(get_tdvector_i(vec_true, i), 0UL) != 0)
		{
			rtd_div(tmp_relerr, tmp_relerr, get_tdvector_i(vec_true, i));
			rtd_abs(tmp_relerr, tmp_relerr);
		}

		if(rtd_cmp(max_relerr, tmp_relerr) < 0)
			rtd_set(max_relerr, tmp_relerr);
		if(rtd_cmp(min_relerr, tmp_relerr) > 0)
			rtd_set(min_relerr, tmp_relerr);
	}

	free_tdvector(tmp_diff_vec);

	return;
}

// relative errors of complex vector
void relerr3_ctdvector(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double max_real_relerr[TDSIZE], double min_real_relerr[TDSIZE], double max_image_relerr[TDSIZE], double min_image_relerr[TDSIZE], double norm_relerr[TDSIZE], CTDVector vec, CTDVector vec_true, int kind_of_norm)
{
	long int i, j, dim;
	tdfloat tmp_relerr[7], tmp_abs;
	double vec_true_norm[TDSIZE];
	ctdfloat ctmp;
	CTDVector tmp_diff_vec;

	dim = vec->re->dim;

	tmp_diff_vec = init_ctdvector(dim);

	// diff_vec := vec - vec_true
	sub_ctdvector(tmp_diff_vec, vec, vec_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_ctdvector(norm_relerr, tmp_diff_vec);
			normi_ctdvector(vec_true_norm, vec_true);
			break;
		case 1: // 1-norm
			norm1_ctdvector(norm_relerr, tmp_diff_vec);
			norm1_ctdvector(vec_true_norm, vec_true);
			break;
		case 2: // 2-norm
		default:
			norm2_ctdvector(norm_relerr, tmp_diff_vec);
			norm2_ctdvector(vec_true_norm, vec_true);
			break;
	}
	if(rtd_cmp_ui(vec_true_norm, 0UL) != 0)
		rtd_div(norm_relerr, norm_relerr, vec_true_norm);

	// relative errors at each elements
	rtd_set_ui(max_relerr, 0UL);
	rtd_set_ui(min_relerr, 0UL);
	rtd_set_ui(max_real_relerr, 0UL);
	rtd_set_ui(min_real_relerr, 0UL);
	rtd_set_ui(max_image_relerr, 0UL);
	rtd_set_ui(min_image_relerr, 0UL);

	for(i = 0; i < dim; i++)
	{
		// Absolute value
		//rctd_abs(&tmp_relerr[0], get_ctdvector_i(tmp_diff_vec, i));
		//rctd_abs(&tmp_abs, get_ctdvector_i(vec_true, i));
		rtd_set(ctmp.val_re, get_tdvector_i(tmp_diff_vec->re, i));
		rtd_set(ctmp.val_im, get_tdvector_i(tmp_diff_vec->im, i));
		rctd_abs(&tmp_relerr[0], &ctmp);
		rtd_set(ctmp.val_re, get_tdvector_i(vec_true->re, i));
		rtd_set(ctmp.val_im, get_tdvector_i(vec_true->im, i));
		rctd_abs(&tmp_abs, &ctmp);

		if(rtd_cmp_ui(tmp_abs.val, 0UL) != 0)
		{
			rtd_div(tmp_relerr[0].val, tmp_relerr[0].val, tmp_abs.val);
			//rtd_abs(tmp_relerr.val, tmp_relerr.val);
		}

		// real part 
		rtd_abs(tmp_relerr[1].val, get_tdvector_i(tmp_diff_vec->re, i));
		rtd_abs(tmp_abs.val, get_tdvector_i(vec_true->re, i));
		if(rtd_cmp_ui(tmp_abs.val, 0UL) != 0)
			rtd_div(tmp_relerr[1].val, tmp_relerr[1].val, tmp_abs.val);

		//rtd_abs(tmp_relerr[1].val, tmp_relerr[1].val);

		// imaginary part 
		rtd_abs(tmp_relerr[2].val, get_tdvector_i(tmp_diff_vec->im, i));
		rtd_abs(tmp_abs.val, get_tdvector_i(vec_true->im, i));
		if(rtd_cmp_ui(tmp_abs.val, 0UL) != 0)
			rtd_div(tmp_relerr[2].val, tmp_relerr[2].val, tmp_abs.val);

		//rtd_abs(tmp_relerr[2].val, tmp_relerr21].val);

		if(rtd_cmp(max_relerr, tmp_relerr[0].val) < 0) rtd_set(max_relerr, tmp_relerr[0].val);
		if(rtd_cmp(min_relerr, tmp_relerr[0].val) > 0) rtd_set(min_relerr, tmp_relerr[0].val);
		if(rtd_cmp(max_real_relerr, tmp_relerr[1].val) < 0) rtd_set(max_real_relerr, tmp_relerr[1].val);
		if(rtd_cmp(min_real_relerr, tmp_relerr[1].val) > 0) rtd_set(min_real_relerr, tmp_relerr[1].val);
		if(rtd_cmp(max_image_relerr, tmp_relerr[2].val) < 0) rtd_set(max_image_relerr, tmp_relerr[2].val);
		if(rtd_cmp(min_image_relerr, tmp_relerr[2].val) > 0) rtd_set(min_image_relerr, tmp_relerr[2].val);
	}

	free_ctdvector(tmp_diff_vec);

	return;
}


// relative errors
void relerr3_tdmatrix(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDMatrix mat, TDMatrix mat_true, int kind_of_norm)
{
	long int i, j, row_dim, col_dim;
	double tmp_relerr[TDSIZE], mat_true_norm[TDSIZE];
	TDMatrix tmp_diff_mat;

	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	tmp_diff_mat = init_tdmatrix(row_dim, col_dim);

	// diff_mat := mat - mat_true
	sub_tdmatrix(tmp_diff_mat, mat, mat_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_tdmatrix(norm_relerr, tmp_diff_mat);
			normi_tdmatrix(mat_true_norm, mat_true);
			break;
		case 1: // 1-norm
			norm1_tdmatrix(norm_relerr, tmp_diff_mat);
			norm1_tdmatrix(mat_true_norm, mat_true);
			break;
		case 2: // Frobenius norm
		default:
			normf_tdmatrix(norm_relerr, tmp_diff_mat);
			normf_tdmatrix(mat_true_norm, mat_true);
			break;
	}
	if(rtd_cmp_ui(mat_true_norm, 0UL) != 0)
		rtd_div(norm_relerr, norm_relerr, mat_true_norm);

	// relative errors at each elements
	rtd_set_ui(max_relerr, 0UL);
	rtd_set_ui(min_relerr, 0UL);

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rtd_abs(tmp_relerr, get_tdmatrix_ij(tmp_diff_mat, i, j));
			if(rtd_cmp_ui(get_tdmatrix_ij(mat_true, i, j), 0UL) != 0)
			{
				rtd_div(tmp_relerr, tmp_relerr, get_tdmatrix_ij(mat_true, i, j));
				rtd_abs(tmp_relerr, tmp_relerr);
			}

			if(rtd_cmp(max_relerr, tmp_relerr) < 0)
				rtd_set(max_relerr, tmp_relerr);
			if(rtd_cmp(min_relerr, tmp_relerr) > 0)
				rtd_set(min_relerr, tmp_relerr);
		}
	}

	free_tdmatrix(tmp_diff_mat);

	return;
}

// relative errors
void relerr3_ctdmatrix(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double max_real_relerr[TDSIZE], double min_real_relerr[TDSIZE], double max_image_relerr[TDSIZE], double min_image_relerr[TDSIZE], double norm_relerr[TDSIZE], CTDMatrix mat, CTDMatrix mat_true, int kind_of_norm)
{
	long int i, j, row_dim, col_dim;
	tdfloat tmp_relerr[3], tmp_abs;
	double mat_true_norm[TDSIZE];
	ctdfloat ctmp;
	CTDMatrix tmp_diff_mat;

	row_dim = mat->re->row_dim;
	col_dim = mat->re->col_dim;

	tmp_diff_mat = init_ctdmatrix(row_dim, col_dim);

	// diff_mat := mat - mat_true
	sub_ctdmatrix(tmp_diff_mat, mat, mat_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_ctdmatrix(norm_relerr, tmp_diff_mat);
			normi_ctdmatrix(mat_true_norm, mat_true);
			break;
		case 1: // 1-norm
			norm1_ctdmatrix(norm_relerr, tmp_diff_mat);
			norm1_ctdmatrix(mat_true_norm, mat_true);
			break;
		case 2: // Frobenius norm
		default:
			normf_ctdmatrix(norm_relerr, tmp_diff_mat);
			normf_ctdmatrix(mat_true_norm, mat_true);
			break;
	}
	if(rtd_cmp_ui(mat_true_norm, 0UL) != 0)
		rtd_div(norm_relerr, norm_relerr, mat_true_norm);

	// relative errors at each elements
	rtd_set_ui(max_relerr, 0UL);
	rtd_set_ui(min_relerr, 0UL);
	rtd_set_ui(max_real_relerr, 0UL);
	rtd_set_ui(min_real_relerr, 0UL);
	rtd_set_ui(max_image_relerr, 0UL);
	rtd_set_ui(min_image_relerr, 0UL);

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			// Absolute value
			//rctd_abs(&tmp_relerr[0], get_ctdmatrix_ij(tmp_diff_mat, i, j));
			rtd_set(ctmp.val_re, get_tdmatrix_ij(tmp_diff_mat->re, i, j));
			rtd_set(ctmp.val_im, get_tdmatrix_ij(tmp_diff_mat->im, i, j));
			rctd_abs(&tmp_relerr[0], &ctmp);
			//rctd_abs(&tmp_abs, get_ctdmatrix_ij(mat_true, i, j));
			rtd_set(ctmp.val_re, get_tdmatrix_ij(mat_true->re, i, j));
			rtd_set(ctmp.val_im, get_tdmatrix_ij(mat_true->im, i, j));
			rctd_abs(&tmp_abs, &ctmp);

			if(rtd_cmp_ui(tmp_abs.val, 0UL) != 0)
			{
				rtd_div(tmp_relerr[0].val, tmp_relerr[0].val, tmp_abs.val); //get_cddmatrix_ij(mat_true, i, j));
				//rtd_abs(tmp_relerr.val, tmp_relerr.val);
			}

			// real part 
			rtd_abs(tmp_relerr[1].val, get_tdmatrix_ij(tmp_diff_mat->re, i, j));
			rtd_abs(tmp_abs.val, get_tdmatrix_ij(mat_true->re, i, j));
			if(rtd_cmp_ui(tmp_abs.val, 0UL) != 0)
				rtd_div(tmp_relerr[1].val, tmp_relerr[1].val, tmp_abs.val);

			//rtd_abs(tmp_relerr[1].val, tmp_relerr[1].val);

			// imaginary part 
			rtd_abs(tmp_relerr[2].val, get_tdmatrix_ij(tmp_diff_mat->im, i, j));
			rtd_abs(tmp_abs.val, get_tdmatrix_ij(mat_true->im, i, j));
			if(rtd_cmp_ui(tmp_abs.val, 0UL) != 0)
				rtd_div(tmp_relerr[2].val, tmp_relerr[2].val, tmp_abs.val);

			//rtd_abs(tmp_relerr[2].val, tmp_relerr21].val);

			if(rtd_cmp(max_relerr, tmp_relerr[0].val) < 0) rtd_set(max_relerr, tmp_relerr[0].val);
			if(rtd_cmp(min_relerr, tmp_relerr[0].val) > 0) rtd_set(min_relerr, tmp_relerr[0].val);
			if(rtd_cmp(max_real_relerr, tmp_relerr[1].val) < 0) rtd_set(max_real_relerr, tmp_relerr[1].val);
			if(rtd_cmp(min_real_relerr, tmp_relerr[1].val) > 0) rtd_set(min_real_relerr, tmp_relerr[1].val);
			if(rtd_cmp(max_image_relerr, tmp_relerr[2].val) < 0) rtd_set(max_image_relerr, tmp_relerr[2].val);
			if(rtd_cmp(min_image_relerr, tmp_relerr[2].val) > 0) rtd_set(min_image_relerr, tmp_relerr[2].val);

		}
	}

	free_ctdmatrix(tmp_diff_mat);

	return;
}

#ifdef USE_GMP
// relative errors for double precision vector
void relerr3_tdvector_mpfvec(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDVector vec, MPFVector vec_true, int kind_of_norm)
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
    subst_mpfvector_tdvec(in_vec, vec);
    relerr3_mpfvector(in_max_relerr, in_min_relerr, in_norm_relerr, in_vec, vec_true, kind_of_norm);

    mpf_get_td(max_relerr, in_max_relerr);
    mpf_get_td(min_relerr, in_min_relerr);
    mpf_get_td(norm_relerr, in_norm_relerr);

    // Free
    mpf_clear(in_max_relerr);
    mpf_clear(in_min_relerr);
    mpf_clear(in_norm_relerr);
    free_mpfvector(in_vec);
}

// relative errors for double precision matrix
void relerr3_tdmatrix_mpfmat(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDMatrix mat, MPFMatrix mat_true, int kind_of_norm)
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
    subst_mpfmatrix_tdmat(in_mat, mat);
    relerr3_mpfmatrix(in_max_relerr, in_min_relerr, in_norm_relerr, in_mat, mat_true, kind_of_norm);

    mpf_get_td(max_relerr, in_max_relerr);
    mpf_get_td(min_relerr, in_min_relerr);
    mpf_get_td(norm_relerr, in_norm_relerr);

    // Free
    mpf_clear(in_max_relerr);
    mpf_clear(in_min_relerr);
    mpf_clear(in_norm_relerr);
    free_mpfmatrix(in_mat);
}

// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Frobenius norm
void relerr3_ctdmatrix_cmpfmat(double max_abs_relerr[TDSIZE], double min_abs_relerr[TDSIZE], double max_real_relerr[TDSIZE], double min_real_relerr[TDSIZE], double max_image_relerr[TDSIZE], double min_image_relerr[TDSIZE], double norm_relerr[TDSIZE], CTDMatrix mat, CMPFMatrix mat_true, int kind_of_norm)
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
    in_mat = init2_cmpfmatrix(mat->re->row_dim, mat->re->col_dim, prec);

    // (mpf_t)in_vec := vec
    subst_cmpfmatrix_ctdmat(in_mat, mat);
    relerr3_cmpfmatrix(in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr, in_mat, mat_true, kind_of_norm);

    mpf_get_td(max_abs_relerr, in_max_abs_relerr);
    mpf_get_td(min_abs_relerr, in_min_abs_relerr);
    mpf_get_td(max_real_relerr, in_max_real_relerr);
    mpf_get_td(min_real_relerr, in_min_real_relerr);
    mpf_get_td(max_image_relerr, in_max_image_relerr);
    mpf_get_td(min_image_relerr, in_min_image_relerr);
    mpf_get_td(norm_relerr, in_norm_relerr);

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
void relerr3_ctdvector_cmpfvec(double max_abs_relerr[TDSIZE], double min_abs_relerr[TDSIZE], double max_real_relerr[TDSIZE], double min_real_relerr[TDSIZE], double max_image_relerr[TDSIZE], double min_image_relerr[TDSIZE], double norm_relerr[TDSIZE], CTDVector vec, CMPFVector vec_true, int kind_of_norm)
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
    in_vec = init2_cmpfvector(vec->re->dim, prec);

    // (mpf_t)in_vec := vec
    subst_cmpfvector_ctdvec(in_vec, vec);
    relerr3_cmpfvector(in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr, in_vec, vec_true, kind_of_norm);

    mpf_get_td(max_abs_relerr, in_max_abs_relerr);
    mpf_get_td(min_abs_relerr, in_min_abs_relerr);
    mpf_get_td(max_real_relerr, in_max_real_relerr);
    mpf_get_td(min_real_relerr, in_min_real_relerr);
    mpf_get_td(max_image_relerr, in_max_image_relerr);
    mpf_get_td(min_image_relerr, in_min_image_relerr);
    mpf_get_td(norm_relerr, in_norm_relerr);

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

#endif // USE_GMP
#endif //USE_TDLINEAR

#ifdef USE_QDLINEAR
// relative errors of vector using higher precision value
void relerr3_tdvector_qdvec(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDVector vec, QDVector vec_true, int kind_of_norm)
{
	long int i, j, dim;
	double tmp_relerr[QDSIZE], vec_true_norm[QDSIZE];
	qdfloat in_ret[7];
	QDVector tmp_diff_vec, in_vec;

	dim = vec->dim;

	tmp_diff_vec= init_qdvector(dim);
	in_vec= init_qdvector(dim); subst_qdvector_tdvec(in_vec, vec);

	// diff_mat := mat - mat_true
	sub_qdvector(tmp_diff_vec, in_vec, vec_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_qdvector(in_ret[0].val, tmp_diff_vec);
			normi_qdvector(in_ret[1].val, vec_true);
			break;
		case 1: // 1-norm
			norm1_qdvector(in_ret[0].val, tmp_diff_vec);
			norm1_qdvector(in_ret[1].val, vec_true);
			break;
		default:
			norm2_qdvector(in_ret[0].val, tmp_diff_vec);
			norm2_qdvector(in_ret[1].val, vec_true);
			break;
	}
	if(rqd_cmp_ui(in_ret[1].val, 0UL) != 0)
		rqd_div(in_ret[0].val, in_ret[0].val, in_ret[1].val);

	// relative errors at each elements
	rqd_set_ui(in_ret[2].val, 0UL);
	rqd_set_ui(in_ret[3].val, 0UL);

	for(i = 0; i < dim; i++)
	{
		rqd_abs(tmp_relerr, get_qdvector_i(tmp_diff_vec, i));
		if(rqd_cmp_ui(get_qdvector_i(vec_true, i), 0UL) != 0)
		{
			rqd_div(tmp_relerr, tmp_relerr, get_qdvector_i(vec_true, i));
			rqd_abs(tmp_relerr, tmp_relerr);
		}

		if(rqd_cmp(in_ret[2].val, tmp_relerr) < 0)
			rqd_set(in_ret[2].val, tmp_relerr);
		if(rqd_cmp(in_ret[3].val, tmp_relerr) > 0)
			rqd_set(in_ret[3].val, tmp_relerr);
	}
	rtd_set_qd(max_relerr, in_ret[2].val);
	rtd_set_qd(min_relerr, in_ret[3].val);
	rtd_set_qd(norm_relerr, in_ret[0].val);

	free_qdvector(tmp_diff_vec);
	free_qdvector(in_vec);

	return;
}

// relative errors of vector
void relerr3_qdvector(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDVector vec, QDVector vec_true, int kind_of_norm)
{
	long int i, j, dim;
	double tmp_relerr[QDSIZE], vec_true_norm[QDSIZE];
	QDVector tmp_diff_vec;

	dim = vec->dim;

	tmp_diff_vec = init_qdvector(dim);

	// diff_mat := mat - mat_true
	sub_qdvector(tmp_diff_vec, vec, vec_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_qdvector(norm_relerr, tmp_diff_vec);
			normi_qdvector(vec_true_norm, vec_true);
			break;
		case 1: // 1-norm
			norm1_qdvector(norm_relerr, tmp_diff_vec);
			norm1_qdvector(vec_true_norm, vec_true);
			break;
		default:
			norm2_qdvector(norm_relerr, tmp_diff_vec);
			norm2_qdvector(vec_true_norm, vec_true);
			break;
	}
	if(rqd_cmp_ui(vec_true_norm, 0UL) != 0)
		rqd_div(norm_relerr, norm_relerr, vec_true_norm);

	// relative errors at each elements
	rqd_set_ui(max_relerr, 0UL);
	rqd_set_ui(min_relerr, 0UL);

	for(i = 0; i < dim; i++)
	{
		rqd_abs(tmp_relerr, get_qdvector_i(tmp_diff_vec, i));
		if(rqd_cmp_ui(get_qdvector_i(vec_true, i), 0UL) != 0)
		{
			rqd_div(tmp_relerr, tmp_relerr, get_qdvector_i(vec_true, i));
			rqd_abs(tmp_relerr, tmp_relerr);
		}

		if(rqd_cmp(max_relerr, tmp_relerr) < 0)
			rqd_set(max_relerr, tmp_relerr);
		if(rqd_cmp(min_relerr, tmp_relerr) > 0)
			rqd_set(min_relerr, tmp_relerr);
	}

	free_qdvector(tmp_diff_vec);

	return;
}

// relative errors of complex vector
void relerr3_cqdvector(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double max_real_relerr[QDSIZE], double min_real_relerr[QDSIZE], double max_image_relerr[QDSIZE], double min_image_relerr[QDSIZE], double norm_relerr[QDSIZE], CQDVector vec, CQDVector vec_true, int kind_of_norm)
{
	long int i, j, dim;
	qdfloat tmp_relerr[7], tmp_abs;
	double vec_true_norm[QDSIZE];
	cqdfloat ctmp;
	CQDVector tmp_diff_vec;

	dim = vec->re->dim;

	tmp_diff_vec = init_cqdvector(dim);

	// diff_vec := vec - vec_true
	sub_cqdvector(tmp_diff_vec, vec, vec_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_cqdvector(norm_relerr, tmp_diff_vec);
			normi_cqdvector(vec_true_norm, vec_true);
			break;
		case 1: // 1-norm
			norm1_cqdvector(norm_relerr, tmp_diff_vec);
			norm1_cqdvector(vec_true_norm, vec_true);
			break;
		case 2: // 2-norm
		default:
			norm2_cqdvector(norm_relerr, tmp_diff_vec);
			norm2_cqdvector(vec_true_norm, vec_true);
			break;
	}
	if(rqd_cmp_ui(vec_true_norm, 0UL) != 0)
		rqd_div(norm_relerr, norm_relerr, vec_true_norm);

	// relative errors at each elements
	rqd_set_ui(max_relerr, 0UL);
	rqd_set_ui(min_relerr, 0UL);
	rqd_set_ui(max_real_relerr, 0UL);
	rqd_set_ui(min_real_relerr, 0UL);
	rqd_set_ui(max_image_relerr, 0UL);
	rqd_set_ui(min_image_relerr, 0UL);

	for(i = 0; i < dim; i++)
	{
		// Absolute value
		//rcqd_abs(&tmp_relerr[0], get_cqdvector_i(tmp_diff_vec, i));
		//rcqd_abs(&tmp_abs, get_cqdvector_i(vec_true, i));
		//rqd_set(ctmp.val_re, get_qdvector_i(tmp_diff_vec->re, i));
		//rqd_set(ctmp.val_im, get_qdvector_i(tmp_diff_vec->im, i));
		subst_cqdvector_i(&ctmp, tmp_diff_vec, i);
		rcqd_abs(&tmp_relerr[0], &ctmp);
		//rqd_set(ctmp.val_re, get_qdvector_i(vec_true->re, i));
		//rqd_set(ctmp.val_im, get_qdvector_i(vec_true->im, i));
		subst_cqdvector_i(&ctmp, vec_true, i);
		rcqd_abs(&tmp_abs, &ctmp);

		if(rqd_cmp_ui(tmp_abs.val, 0UL) != 0)
		{
			rqd_div(tmp_relerr[0].val, tmp_relerr[0].val, tmp_abs.val);
			//rqd_abs(tmp_relerr.val, tmp_relerr.val);
		}

		// real part 
		rqd_abs(tmp_relerr[1].val, get_qdvector_i(tmp_diff_vec->re, i));
		rqd_abs(tmp_abs.val, get_qdvector_i(vec_true->re, i));
		if(rqd_cmp_ui(tmp_abs.val, 0UL) != 0)
			rqd_div(tmp_relerr[1].val, tmp_relerr[1].val, tmp_abs.val);

		//rtd_abs(tmp_relerr[1].val, tmp_relerr[1].val);

		// imaginary part 
		rqd_abs(tmp_relerr[2].val, get_qdvector_i(tmp_diff_vec->im, i));
		rqd_abs(tmp_abs.val, get_qdvector_i(vec_true->im, i));
		if(rqd_cmp_ui(tmp_abs.val, 0UL) != 0)
			rqd_div(tmp_relerr[2].val, tmp_relerr[2].val, tmp_abs.val);

		//rqd_abs(tmp_relerr[2].val, tmp_relerr21].val);

		if(rqd_cmp(max_relerr, tmp_relerr[0].val) < 0) rqd_set(max_relerr, tmp_relerr[0].val);
		if(rqd_cmp(min_relerr, tmp_relerr[0].val) > 0) rqd_set(min_relerr, tmp_relerr[0].val);
		if(rqd_cmp(max_real_relerr, tmp_relerr[1].val) < 0) rqd_set(max_real_relerr, tmp_relerr[1].val);
		if(rqd_cmp(min_real_relerr, tmp_relerr[1].val) > 0) rqd_set(min_real_relerr, tmp_relerr[1].val);
		if(rqd_cmp(max_image_relerr, tmp_relerr[2].val) < 0) rqd_set(max_image_relerr, tmp_relerr[2].val);
		if(rqd_cmp(min_image_relerr, tmp_relerr[2].val) > 0) rqd_set(min_image_relerr, tmp_relerr[2].val);
	}

	free_cqdvector(tmp_diff_vec);

	return;
}

// relative errors
void relerr3_qdmatrix(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDMatrix mat, QDMatrix mat_true, int kind_of_norm)
{
	long int i, j, row_dim, col_dim;
	double tmp_relerr[QDSIZE], mat_true_norm[QDSIZE];
	QDMatrix tmp_diff_mat;

	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	tmp_diff_mat = init_qdmatrix(row_dim, col_dim);

	// diff_mat := mat - mat_true
	sub_qdmatrix(tmp_diff_mat, mat, mat_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_qdmatrix(norm_relerr, tmp_diff_mat);
			normi_qdmatrix(mat_true_norm, mat_true);
			break;
		case 1: // 1-norm
			norm1_qdmatrix(norm_relerr, tmp_diff_mat);
			norm1_qdmatrix(mat_true_norm, mat_true);
			break;
		case 2: // Frobenius norm
		default:
			normf_qdmatrix(norm_relerr, tmp_diff_mat);
			normf_qdmatrix(mat_true_norm, mat_true);
			break;
	}
	if(rqd_cmp_ui(mat_true_norm, 0UL) != 0)
		rqd_div(norm_relerr, norm_relerr, mat_true_norm);

	// relative errors at each elements
	rqd_set_ui(max_relerr, 0UL);
	rqd_set_ui(min_relerr, 0UL);

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rqd_abs(tmp_relerr, get_qdmatrix_ij(tmp_diff_mat, i, j));
			if(rqd_cmp_ui(get_qdmatrix_ij(mat_true, i, j), 0UL) != 0)
			{
				rqd_div(tmp_relerr, tmp_relerr, get_qdmatrix_ij(mat_true, i, j));
				rqd_abs(tmp_relerr, tmp_relerr);
			}

			if(rqd_cmp(max_relerr, tmp_relerr) < 0)
				rqd_set(max_relerr, tmp_relerr);
			if(rqd_cmp(min_relerr, tmp_relerr) > 0)
				rqd_set(min_relerr, tmp_relerr);
		}
	}

	free_qdmatrix(tmp_diff_mat);

	return;
}

// relative errors
void relerr3_cqdmatrix(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double max_real_relerr[QDSIZE], double min_real_relerr[QDSIZE], double max_image_relerr[QDSIZE], double min_image_relerr[QDSIZE], double norm_relerr[QDSIZE], CQDMatrix mat, CQDMatrix mat_true, int kind_of_norm)
{
	long int i, j, row_dim, col_dim;
	qdfloat tmp_relerr[3], tmp_abs;
	double mat_true_norm[QDSIZE];
	cqdfloat ctmp;
	CQDMatrix tmp_diff_mat;

	row_dim = mat->re->row_dim;
	col_dim = mat->re->col_dim;

	tmp_diff_mat = init_cqdmatrix(row_dim, col_dim);

	// diff_mat := mat - mat_true
	sub_cqdmatrix(tmp_diff_mat, mat, mat_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_cqdmatrix(norm_relerr, tmp_diff_mat);
			normi_cqdmatrix(mat_true_norm, mat_true);
			break;
		case 1: // 1-norm
			norm1_cqdmatrix(norm_relerr, tmp_diff_mat);
			norm1_cqdmatrix(mat_true_norm, mat_true);
			break;
		case 2: // Frobenius norm
		default:
			normf_cqdmatrix(norm_relerr, tmp_diff_mat);
			normf_cqdmatrix(mat_true_norm, mat_true);
			break;
	}
	if(rqd_cmp_ui(mat_true_norm, 0UL) != 0)
		rqd_div(norm_relerr, norm_relerr, mat_true_norm);

	// relative errors at each elements
	rqd_set_ui(max_relerr, 0UL);
	rqd_set_ui(min_relerr, 0UL);
	rqd_set_ui(max_real_relerr, 0UL);
	rqd_set_ui(min_real_relerr, 0UL);
	rqd_set_ui(max_image_relerr, 0UL);
	rqd_set_ui(min_image_relerr, 0UL);

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			// Absolute value
			//rcqd_abs(&tmp_relerr[0], get_cqdmatrix_ij(tmp_diff_mat, i, j));
			//rqd_set(ctmp.val_re, get_qdmatrix_ij(tmp_diff_mat->re, i, j));
			//rqd_set(ctmp.val_im, get_qdmatrix_ij(tmp_diff_mat->im, i, j));
			subst_cqdmatrix_ij(&ctmp, tmp_diff_mat, i, j);
			rcqd_abs(&tmp_relerr[0], &ctmp);
			//rcqd_abs(&tmp_abs, get_cqdmatrix_ij(mat_true, i, j));
			//rqd_set(ctmp.val_re, get_qdmatrix_ij(mat_true->re, i, j));
			//rqd_set(ctmp.val_im, get_qdmatrix_ij(mat_true->im, i, j));
			subst_cqdmatrix_ij(&ctmp, mat_true, i, j);
			rcqd_abs(&tmp_abs, &ctmp);

			if(rqd_cmp_ui(tmp_abs.val, 0UL) != 0)
			{
				rqd_div(tmp_relerr[0].val, tmp_relerr[0].val, tmp_abs.val); //get_cddmatrix_ij(mat_true, i, j));
				//rtd_abs(tmp_relerr.val, tmp_relerr.val);
			}

			// real part
			rqd_abs(tmp_relerr[1].val, get_qdmatrix_ij(tmp_diff_mat->re, i, j));
			rqd_abs(tmp_abs.val, get_qdmatrix_ij(mat_true->re, i, j));
			if(rqd_cmp_ui(tmp_abs.val, 0UL) != 0)
				rqd_div(tmp_relerr[1].val, tmp_relerr[1].val, tmp_abs.val);

			//rtd_abs(tmp_relerr[1].val, tmp_relerr[1].val);

			// imaginary part 
			rqd_abs(tmp_relerr[2].val, get_qdmatrix_ij(tmp_diff_mat->im, i, j));
			rqd_abs(tmp_abs.val, get_qdmatrix_ij(mat_true->im, i, j));
			if(rqd_cmp_ui(tmp_abs.val, 0UL) != 0)
				rqd_div(tmp_relerr[2].val, tmp_relerr[2].val, tmp_abs.val);

			//rtd_abs(tmp_relerr[2].val, tmp_relerr21].val);

			if(rqd_cmp(max_relerr, tmp_relerr[0].val) < 0) rqd_set(max_relerr, tmp_relerr[0].val);
			if(rqd_cmp(min_relerr, tmp_relerr[0].val) > 0) rqd_set(min_relerr, tmp_relerr[0].val);
			if(rqd_cmp(max_real_relerr, tmp_relerr[1].val) < 0) rqd_set(max_real_relerr, tmp_relerr[1].val);
			if(rqd_cmp(min_real_relerr, tmp_relerr[1].val) > 0) rqd_set(min_real_relerr, tmp_relerr[1].val);
			if(rqd_cmp(max_image_relerr, tmp_relerr[2].val) < 0) rqd_set(max_image_relerr, tmp_relerr[2].val);
			if(rqd_cmp(min_image_relerr, tmp_relerr[2].val) > 0) rqd_set(min_image_relerr, tmp_relerr[2].val);

		}
	}

	free_cqdmatrix(tmp_diff_mat);

	return;
}

#ifdef USE_GMP
// relative errors for double precision vector
void relerr3_qdvector_mpfvec(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDVector vec, MPFVector vec_true, int kind_of_norm)
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
    subst_mpfvector_qdvec(in_vec, vec);
    relerr3_mpfvector(in_max_relerr, in_min_relerr, in_norm_relerr, in_vec, vec_true, kind_of_norm);

    mpf_get_qd(max_relerr, in_max_relerr);
    mpf_get_qd(min_relerr, in_min_relerr);
    mpf_get_qd(norm_relerr, in_norm_relerr);

    // Free
    mpf_clear(in_max_relerr);
    mpf_clear(in_min_relerr);
    mpf_clear(in_norm_relerr);
    free_mpfvector(in_vec);
}

// relative errors for double precision matrix
void relerr3_qdmatrix_mpfmat(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDMatrix mat, MPFMatrix mat_true, int kind_of_norm)
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
    subst_mpfmatrix_qdmat(in_mat, mat);
    relerr3_mpfmatrix(in_max_relerr, in_min_relerr, in_norm_relerr, in_mat, mat_true, kind_of_norm);

    mpf_get_qd(max_relerr, in_max_relerr);
    mpf_get_qd(min_relerr, in_min_relerr);
    mpf_get_qd(norm_relerr, in_norm_relerr);

    // Free
    mpf_clear(in_max_relerr);
    mpf_clear(in_min_relerr);
    mpf_clear(in_norm_relerr);
    free_mpfmatrix(in_mat);
}

// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Frobenius norm
void relerr3_cqdmatrix_cmpfmat(double max_abs_relerr[QDSIZE], double min_abs_relerr[QDSIZE], double max_real_relerr[QDSIZE], double min_real_relerr[QDSIZE], double max_image_relerr[QDSIZE], double min_image_relerr[QDSIZE], double norm_relerr[QDSIZE], CQDMatrix mat, CMPFMatrix mat_true, int kind_of_norm)
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
    in_mat = init2_cmpfmatrix(mat->re->row_dim, mat->re->col_dim, prec);

    // (mpf_t)in_vec := vec
    subst_cmpfmatrix_cqdmat(in_mat, mat);
    relerr3_cmpfmatrix(in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr, in_mat, mat_true, kind_of_norm);

    mpf_get_qd(max_abs_relerr, in_max_abs_relerr);
    mpf_get_qd(min_abs_relerr, in_min_abs_relerr);
    mpf_get_qd(max_real_relerr, in_max_real_relerr);
    mpf_get_qd(min_real_relerr, in_min_real_relerr);
    mpf_get_qd(max_image_relerr, in_max_image_relerr);
    mpf_get_qd(min_image_relerr, in_min_image_relerr);
    mpf_get_qd(norm_relerr, in_norm_relerr);

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
void relerr3_cqdvector_cmpfvec(double max_abs_relerr[QDSIZE], double min_abs_relerr[QDSIZE], double max_real_relerr[QDSIZE], double min_real_relerr[QDSIZE], double max_image_relerr[QDSIZE], double min_image_relerr[QDSIZE], double norm_relerr[QDSIZE], CQDVector vec, CMPFVector vec_true, int kind_of_norm)
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
    in_vec = init2_cmpfvector(vec->re->dim, prec);

    // (mpf_t)in_vec := vec
    subst_cmpfvector_cqdvec(in_vec, vec);
    relerr3_cmpfvector(in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr, in_vec, vec_true, kind_of_norm);

    mpf_get_qd(max_abs_relerr, in_max_abs_relerr);
    mpf_get_qd(min_abs_relerr, in_min_abs_relerr);
    mpf_get_qd(max_real_relerr, in_max_real_relerr);
    mpf_get_qd(min_real_relerr, in_min_real_relerr);
    mpf_get_qd(max_image_relerr, in_max_image_relerr);
    mpf_get_qd(min_image_relerr, in_min_image_relerr);
    mpf_get_qd(norm_relerr, in_norm_relerr);

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

#endif // USE_GMP

#endif // USE_QDLINEAR
#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus
