//#include "bnc.h"
#include "matmul_strassen.h"

#ifdef USE_OMP
	#include "bncomp.h"
#endif // USE_OMP

// DD and QD
//#include "ddlinear.h"

#ifdef USE_DDLINEAR

// ANSI C
#ifndef __cplusplus

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

#ifdef USE_TDLINEAR

// relative errors
void relerr3_tdmatrix(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], TDMatrix mat, TDMatrix mat_true, int kind_of_norm)
{
	long int i, j, row_dim, col_dim;
	double tmp_relerr[QDSIZE], mat_true_norm[QDSIZE];
	QDMatrix tmp_diff_mat, qd_mat_true;

	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	tmp_diff_mat = init_qdmatrix(row_dim, col_dim);
	qd_mat_true  = init_qdmatrix(row_dim, col_dim);
	subst_qdmatrix_tdmat(qd_mat_true, mat_true);

	// diff_mat := mat - mat_true
	sub_qdmatrix_tdmat_tdmat(tmp_diff_mat, mat, mat_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_qdmatrix(norm_relerr, tmp_diff_mat);
			normi_qdmatrix(mat_true_norm, qd_mat_true);
			break;
		case 1: // 1-norm
			norm1_qdmatrix(norm_relerr, tmp_diff_mat);
			norm1_qdmatrix(mat_true_norm, qd_mat_true);
			break;
		case 2: // Frobenius norm
		default:
			normf_qdmatrix(norm_relerr, tmp_diff_mat);
			normf_qdmatrix(mat_true_norm, qd_mat_true);
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
			if(rqd_cmp_ui(get_qdmatrix_ij(qd_mat_true, i, j), 0UL) != 0)
			{
				rqd_div(tmp_relerr, tmp_relerr, get_qdmatrix_ij(qd_mat_true, i, j));
				rqd_abs(tmp_relerr, tmp_relerr);
			}

			if(rqd_cmp(max_relerr, tmp_relerr) < 0)
				rqd_set(max_relerr, tmp_relerr);
			if(rqd_cmp(min_relerr, tmp_relerr) > 0)
				rqd_set(min_relerr, tmp_relerr);
		}
	}

	free_qdmatrix(tmp_diff_mat);
	free_qdmatrix(qd_mat_true);

	return;
}
#endif //USE_TDLINEAR

#ifdef USE_QDLINEAR

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

// C++
#else // __cplusplus

// relative errors
void relerr3_ddmatrix(dd_real *max_relerr, dd_real *min_relerr, dd_real *norm_relerr, DDMatrix mat, DDMatrix mat_true, int kind_of_norm)
{
	long int i, j, row_dim, col_dim;
	dd_real tmp_relerr, mat_true_norm;
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
			normi_ddmatrix(&mat_true_norm, mat_true);
			break;
		case 1: // 1-norm
			norm1_ddmatrix(norm_relerr, tmp_diff_mat);
			norm1_ddmatrix(&mat_true_norm, mat_true);
			break;
		case 2: // Frobenius norm
		default:
			normf_ddmatrix(norm_relerr, tmp_diff_mat);
			normf_ddmatrix(&mat_true_norm, mat_true);
			break;
	}
	if(mat_true_norm != 0.0)
		*norm_relerr = *norm_relerr / mat_true_norm;

	// relative errors at each elements
	rdd_set_ui(*max_relerr, 0UL);
	rdd_set_ui(*min_relerr, 0UL);

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rdd_abs(tmp_relerr, get_ddmatrix_ij(tmp_diff_mat, i, j));
			if(get_ddmatrix_ij(mat_true, i, j) != 0.0)
			{
				rdd_div(tmp_relerr, tmp_relerr, get_ddmatrix_ij(mat_true, i, j));
				rdd_abs(tmp_relerr, tmp_relerr);
			}

			if(rdd_cmp(*max_relerr, tmp_relerr) < 0)
				rdd_set(*max_relerr, tmp_relerr);
			if(rdd_cmp(*min_relerr, tmp_relerr) > 0)
				rdd_set(*min_relerr, tmp_relerr);
		}
	}

	free_ddmatrix(tmp_diff_mat);

	return;
}

// relative errors
//void relerr3_tdmatrix(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDMatrix mat, TDMatrix mat_true, int kind_of_norm)
void relerr3_tdmatrix(qd_real *max_relerr, qd_real *min_relerr, qd_real *norm_relerr, TDMatrix mat, TDMatrix mat_true, int kind_of_norm)
{
	long int i, j, row_dim, col_dim;
	qd_real tmp_relerr, mat_true_norm;
	QDMatrix tmp_diff_mat, qd_mat_true;

	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	tmp_diff_mat = init_qdmatrix(row_dim, col_dim);
	qd_mat_true  = init_qdmatrix(row_dim, col_dim);
	subst_qdmatrix_tdmat(qd_mat_true, mat_true);

	// diff_mat := mat - mat_true
	sub_qdmatrix_tdmat_tdmat(tmp_diff_mat, mat, mat_true);

	// norm_relerr
	switch(kind_of_norm)
	{
		case 0: // infinit norm
			normi_qdmatrix(norm_relerr, tmp_diff_mat);
			normi_qdmatrix(&mat_true_norm, qd_mat_true);
			break;
		case 1: // 1-norm
			norm1_qdmatrix(norm_relerr, tmp_diff_mat);
			norm1_qdmatrix(&mat_true_norm, qd_mat_true);
			break;
		case 2: // Frobenius norm
		default:
			normf_qdmatrix(norm_relerr, tmp_diff_mat);
			normf_qdmatrix(&mat_true_norm, qd_mat_true);
			break;
	}
	if(mat_true_norm != 0.0)
		*norm_relerr = *norm_relerr / mat_true_norm;

	// relative errors at each elements
	rqd_set_ui(*max_relerr, 0UL);
	rqd_set_ui(*min_relerr, 0UL);

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rqd_abs(tmp_relerr, get_qdmatrix_ij(tmp_diff_mat, i, j));
			if(get_qdmatrix_ij(qd_mat_true, i, j) != 0.0)
			{
				rqd_div(tmp_relerr, tmp_relerr, get_qdmatrix_ij(qd_mat_true, i, j));
				rqd_abs(tmp_relerr, tmp_relerr);
			}

			if(rqd_cmp(*max_relerr, tmp_relerr) < 0)
				rqd_set(*max_relerr, tmp_relerr);
			if(rqd_cmp(*min_relerr, tmp_relerr) > 0)
				rqd_set(*min_relerr, tmp_relerr);
		}
	}

	free_qdmatrix(tmp_diff_mat);
	free_qdmatrix(qd_mat_true);

	return;
}


// relative errors
void relerr3_qdmatrix(qd_real *max_relerr, qd_real *min_relerr, qd_real *norm_relerr, QDMatrix mat, QDMatrix mat_true, int kind_of_norm)
{
	long int i, j, row_dim, col_dim;
	qd_real tmp_relerr, mat_true_norm;
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
			normi_qdmatrix(&mat_true_norm, mat_true);
			break;
		case 1: // 1-norm
			norm1_qdmatrix(norm_relerr, tmp_diff_mat);
			norm1_qdmatrix(&mat_true_norm, mat_true);
			break;
		case 2: // Frobenius norm
		default:
			normf_qdmatrix(norm_relerr, tmp_diff_mat);
			normf_qdmatrix(&mat_true_norm, mat_true);
			break;
	}
	if(mat_true_norm != 0.0)
		*norm_relerr = *norm_relerr / mat_true_norm;

	// relative errors at each elements
	rqd_set_ui(*max_relerr, 0UL);
	rqd_set_ui(*min_relerr, 0UL);

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rqd_abs(tmp_relerr, get_qdmatrix_ij(tmp_diff_mat, i, j));
			if(get_qdmatrix_ij(mat_true, i, j) != 0.0)
			{
				rqd_div(tmp_relerr, tmp_relerr, get_qdmatrix_ij(mat_true, i, j));
				rqd_abs(tmp_relerr, tmp_relerr);
			}

			if(rqd_cmp(*max_relerr, tmp_relerr) < 0)
				rqd_set(*max_relerr, tmp_relerr);
			if(rqd_cmp(*min_relerr, tmp_relerr) > 0)
				rqd_set(*min_relerr, tmp_relerr);
		}
	}

	free_qdmatrix(tmp_diff_mat);

	return;
}

//#endif // __cplusplus

#endif // USE_DDLINEAR
