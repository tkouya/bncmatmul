/********************************************************************************/
/* td_oz_scheme: Multiple precision linear computation based on Ozaki scheme.   */
/* Copyright (C) 2022 Tomonori Kouya                                            */
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
#include "rdd.h" // [dtq]dfloat and its arithmetic r[dtq]d_*
#include "matmul_strassen.h"

// absmax_tdvector
void absmax_tdvector(double ret[TDSIZE], long int *max_index, TDVector vec)
{
    long int i, max_i, dim = vec->dim;
    double abs_val[TDSIZE];

    max_i = 0;
    ret[0] = 0.0; ret[1] = 0.0;
    for(i = 0; i < dim; i++)
    {
        //abs_val = fabs(get_tdvector_i(vec, i));
        rtd_abs(abs_val, get_tdvector_i(vec, i));
        if(rtd_cmp(ret, abs_val) < 0)
        {
            rtd_set(ret, abs_val);
            max_i = i;
        }
    }

    if(max_index != NULL)
        *max_index = max_i;

    return;
}

/* c = a + (double)b */
void add_tdvector_dvec(TDVector c, TDVector a, DVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_tdvector_dvec\n");
		return;
	}

	double tmp[TDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rtd_add_d(tmp, get_tdvector_i(a, i), get_dvector_i(b, i));
		set_tdvector_i(c, i, tmp);
	}
}

/* c = a - (double)b */
void sub_tdvector_dvec(TDVector c, TDVector a, DVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_tdvector_dvec\n");
		return;
	}

	double tmp[TDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rtd_sub_d(tmp, get_tdvector_i(a, i),  get_dvector_i(b, i));
		set_tdvector_i(c, i, tmp);
	}
}

// extract vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
//void extract_tdvector(DVector ret_vec[], int num_div, TDVector org_vec, int num_bits)
/*------------------------------------------------------------------------------*/
/* split vector: 2^shift[index] * (ret_vec[0] + ... ) = org_vec                  */
/*                                                                               */
/* The vector plays the role of a single column of B, so it carries one exponent */
/* per slice.  shift may be NULL; see oz_scheme.h.                               */
/*------------------------------------------------------------------------------*/
int split_tdvector_dvec_ex(DVector ret_vec[], long int shift[], int num_div, TDVector org_vec)
{
	long int dim = org_vec->dim, i;
	int index, num_bits = 53, real_num_div; // IEEE754 binary64
	double absmax_org_vec, threshold, t_exp, tail_exp, org_vec_i, high_i;
	double org_ii[TDSIZE], high_td[TDSIZE], rest_i[TDSIZE];
	long int sigma;
	TDVector tmp_org_vec;
	DVector own_ret_vec = NULL, in_ret_vec;
	int comp;

	if(ret_vec == NULL)
	{
		own_ret_vec = init_dvector(dim);
		in_ret_vec = own_ret_vec;
	}
	else
		in_ret_vec = ret_vec[0];

	// tmp_org_vec := org_vec
	tmp_org_vec = init_tdvector(dim);
	subst_tdvector(tmp_org_vec, org_vec);

	tail_exp = ceil(((double)num_bits + DLOG2((double)(dim))) / 2.0);

	for(comp = 0; comp < TDSIZE; comp++)
		high_td[comp] = 0.0;

	real_num_div = 0;
	for(index = 0; index < num_div; index++)
	{
		if(ret_vec != NULL)
			in_ret_vec = ret_vec[index];

		absmax_org_vec = 0.0;
		for(i = 0; i < dim; i++)
		{
			org_vec_i = fabs(tmp_org_vec->element[0][i]);
			if(org_vec_i > absmax_org_vec)
				absmax_org_vec = org_vec_i;
		}

		// vector is exhausted
		if(absmax_org_vec == 0.0) break;

		sigma = 0;
		if(shift != NULL)
		{
			sigma = bnc_oz_exp2_d(absmax_org_vec);
			if(sigma < BNC_OZ_MIN_SCALED_EXP)
				sigma = 0;
			shift[index] = sigma;

			if(sigma != 0)
				absmax_org_vec = bnc_oz_ldexp(absmax_org_vec, -sigma);
		}

		t_exp = ceil(DLOG2(absmax_org_vec)) + tail_exp;
		threshold = pow(2.0, t_exp);

		for(i = 0; i < dim; i++)
		{
			org_vec_i = tmp_org_vec->element[0][i];
			if(sigma != 0)
				org_vec_i = bnc_oz_ldexp(org_vec_i, -sigma);

			high_i = org_vec_i + threshold;
			high_i = high_i - threshold;
			set_dvector_i(in_ret_vec, i, high_i);

			for(comp = 0; comp < TDSIZE; comp++)
				org_ii[comp] = tmp_org_vec->element[comp][i];
			high_td[0] = (sigma != 0) ? bnc_oz_ldexp(high_i, sigma) : high_i;

			rtd_sub(rest_i, org_ii, high_td);

			for(comp = 0; comp < TDSIZE; comp++)
				tmp_org_vec->element[comp][i] = rest_i[comp];
		}

		real_num_div = index + 1;
	}

	free_tdvector(tmp_org_vec);
	if(own_ret_vec != NULL)
		free_dvector(own_ret_vec);

	return real_num_div;
}

// split vector without the scaling
int split_tdvector_dvec(DVector ret_vec[], int num_div, TDVector org_vec)
{
	return split_tdvector_dvec_ex(ret_vec, NULL, num_div, org_vec);
}

// absmax_row_tdmatrix
void absmax_row_tdmatrix(double mu[TDSIZE], long int *max_j, long int row_index, TDMatrix mat)
{
    long int j, max_index = 0;
    double abs_aij[TDSIZE];

	//mu = fabs(mat[i * col_dim + 0]);
    rtd_abs(mu, get_tdmatrix_ij(mat, row_index, 0));

	for(j = 1; j < mat->col_dim; j++)
	{
		rtd_abs(abs_aij, get_tdmatrix_ij(mat, row_index, j));
		if(rtd_cmp(abs_aij, mu) > 0)
        {
			//mu = abs_aij;
            rtd_set(mu, abs_aij);
            max_index = j;
        }
	}

    if(max_j != NULL)
        *max_j = max_index;

    //return mu;
    return;
}

/* c := a + (doble)b */
void add_tdmatrix_dmat(TDMatrix c, TDMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim, a_stride, b_stride, c_stride;
	int num_threads;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_tdmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_tdmatrix_dmat\n");
		return;
	}
	col_dim = c->col_dim;

	a_stride = a->real_col_dim;
	b_stride = b->real_col_dim;
	c_stride = c->real_col_dim;

	num_threads = bnc_oz_get_num_threads();

	// row-wise: the rows are independent, so this is the whole parallelization
#ifdef _OPENMP
	#pragma omp parallel for num_threads(num_threads) schedule(static) private(i, j)
#endif // _OPENMP
	for(i = 0; i < row_dim; i++)
	{
		const double *a_row[TDSIZE];
		const double *b_row = b->element + i * b_stride;
		double *c_row[TDSIZE];
		double a_ij[TDSIZE], b_ij[TDSIZE], c_ij[TDSIZE];
		int comp;

		for(comp = 0; comp < TDSIZE; comp++)
		{
			a_row[comp] = a->element[comp] + i * a_stride;
			c_row[comp] = c->element[comp] + i * c_stride;
			b_ij[comp] = 0.0;
		}

		for(j = 0; j < col_dim; j++)
		{
			for(comp = 0; comp < TDSIZE; comp++)
				a_ij[comp] = a_row[comp][j];
			b_ij[0] = b_row[j];

			rtd_add(c_ij, a_ij, b_ij);

			for(comp = 0; comp < TDSIZE; comp++)
				c_row[comp][j] = c_ij[comp];
		}
	}
}

/* c := a - (doble)b */
void sub_tdmatrix_dmat(TDMatrix c, TDMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim, a_stride, b_stride, c_stride;
	int num_threads;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_tdmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_tdmatrix_dmat\n");
		return;
	}
	col_dim = c->col_dim;

	a_stride = a->real_col_dim;
	b_stride = b->real_col_dim;
	c_stride = c->real_col_dim;

	num_threads = bnc_oz_get_num_threads();

	// row-wise: the rows are independent, so this is the whole parallelization
#ifdef _OPENMP
	#pragma omp parallel for num_threads(num_threads) schedule(static) private(i, j)
#endif // _OPENMP
	for(i = 0; i < row_dim; i++)
	{
		const double *a_row[TDSIZE];
		const double *b_row = b->element + i * b_stride;
		double *c_row[TDSIZE];
		double a_ij[TDSIZE], b_ij[TDSIZE], c_ij[TDSIZE];
		int comp;

		for(comp = 0; comp < TDSIZE; comp++)
		{
			a_row[comp] = a->element[comp] + i * a_stride;
			c_row[comp] = c->element[comp] + i * c_stride;
			b_ij[comp] = 0.0;
		}

		for(j = 0; j < col_dim; j++)
		{
			for(comp = 0; comp < TDSIZE; comp++)
				a_ij[comp] = a_row[comp][j];
			b_ij[0] = b_row[j];

			rtd_sub(c_ij, a_ij, b_ij);

			for(comp = 0; comp < TDSIZE; comp++)
				c_row[comp][j] = c_ij[comp];
		}
	}
}

//#define SPLIT_NUM_DIGITS 64

// SplitMat_A
/*------------------------------------------------------------------------------*/
/* SplitMat_A: ret_mat[0] + ret_mat[1] + ... = org_mat, every ret_mat[] a plain  */
/* double matrix whose rows multiply without rounding.                           */
/*                                                                               */
/* Two passes over the data per split instead of the six the straightforward     */
/* formulation needs (copy, row maximum, fill s, mat + s, - s, mat - high), and  */
/* both passes run row-parallel because rows never interact.  The threshold s is */
/* constant along a row, so a vector of row thresholds replaces the full s       */
/* matrix that used to be built and streamed for every split.                    */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
/* SplitMat_A: 2^row_shift[index][i] * (ret_mat[0] + ret_mat[1] + ...) = org_mat,*/
/* every ret_mat[] a plain double matrix whose rows multiply without rounding.   */
/*                                                                               */
/* Two passes over the data per split instead of the six the straightforward     */
/* formulation needs, and both run row-parallel because rows never interact.     */
/* The threshold is constant along a row, so a vector of row thresholds replaces */
/* the full s matrix that used to be built and streamed for every split, and the */
/* row is normalized first so that neither the threshold nor the slice can leave */
/* the double exponent range.  row_shift may be NULL; see oz_scheme.h.           */
/*------------------------------------------------------------------------------*/
int split_tdmatrix_dmat_ex(DMatrix ret_mat[], long int row_shift[], int num_div, TDMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, org_stride, ret_stride;
	long int num_digits = 53; // IEEE double prec.
	int real_num_div, num_threads;
	double mu_total, tail_exp;
	TDMatrix tmp_org_mat;
	DMatrix own_ret_mat = NULL, in_ret_mat;
	double *power2; // 2^t_exp of each row, in the scaled domain
	long int *shift;

	row_dim = org_mat->row_dim;
	col_dim = org_mat->col_dim;

	if(ret_mat != NULL)
	{
		if((ret_mat[0]->row_dim != row_dim) || (ret_mat[0]->col_dim != col_dim))
		{
			fprintf(stderr, "ERROR: split_tdmatrix_dmat\n");
			return 0;
		}
	}

	power2 = (double *)calloc((size_t)row_dim, sizeof(double));
	if(power2 == NULL)
	{
		fprintf(stderr, "ERROR: split_tdmatrix_dmat: cannot allocate\n");
		return 0;
	}

	if(ret_mat == NULL)
	{
		own_ret_mat = init_dmatrix(row_dim, col_dim);
		in_ret_mat = own_ret_mat;
	}
	else
		in_ret_mat = ret_mat[0];

	// tmp_org_mat := org_mat; it always holds the part not split off yet
	tmp_org_mat = init_tdmatrix(row_dim, col_dim);
	subst_tdmatrix(tmp_org_mat, org_mat);
	org_stride = tmp_org_mat->real_col_dim;

	// the row-independent half of t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim) / 2)
	tail_exp = ceil(((double)num_digits + DLOG2((double)col_dim)) / 2.0);

	num_threads = bnc_oz_get_num_threads();

	real_num_div = 0;
	for(index = 0; index < num_div; index++)
	{
		if(ret_mat != NULL)
			in_ret_mat = ret_mat[index];
		ret_stride = in_ret_mat->real_col_dim;
		shift = (row_shift != NULL) ? (row_shift + (size_t)index * (size_t)row_dim) : NULL;

		// pass 1: ret_mat[index] := 2^-shift[i] * tmp_org_mat, mu[i] = max_j |ret_mat[index][i][j]|
		mu_total = 0.0;

#ifdef _OPENMP
		#pragma omp parallel for num_threads(num_threads) schedule(static) private(i, j) reduction(+:mu_total)
#endif // _OPENMP
		for(i = 0; i < row_dim; i++)
		{
			const double *org_row = tmp_org_mat->element[0] + i * org_stride;
			double *ret_row = in_ret_mat->element + i * ret_stride;
			double mu = 0.0, abs_org_ij;
			long int sigma = 0;

			for(j = 0; j < col_dim; j++)
			{
				abs_org_ij = fabs(org_row[j]); // == |rtd_get_d(org[i][j])|
				if(abs_org_ij > mu)
					mu = abs_org_ij;
			}

			if(shift != NULL)
			{
				sigma = bnc_oz_exp2_d(mu);
				if(sigma < BNC_OZ_MIN_SCALED_EXP)
					sigma = 0; // shifting back would not be exact down there
				shift[i] = sigma;
			}

			if(sigma != 0)
			{
				for(j = 0; j < col_dim; j++)
					ret_row[j] = bnc_oz_ldexp(org_row[j], -sigma);

				mu = bnc_oz_ldexp(mu, -sigma);
			}
			else
			{
				for(j = 0; j < col_dim; j++)
					ret_row[j] = org_row[j];
			}

			// s[i, j] = 2^t_exp
			power2[i] = pow(2.0, ceil(DLOG2(mu)) + tail_exp);
			mu_total += mu;
		}

		// nothing left to split
		if(mu_total == 0.0) break;

		// pass 2: high := (mat + s) - s, and mat := mat - 2^shift[i] * high
#ifdef _OPENMP
		#pragma omp parallel for num_threads(num_threads) schedule(static) private(i, j)
#endif // _OPENMP
		for(i = 0; i < row_dim; i++)
		{
			double *ret_row = in_ret_mat->element + i * ret_stride;
			double *org_row[TDSIZE];
			double s = power2[i], high_ij;
			double org_ij[TDSIZE], high_td[TDSIZE], rest_ij[TDSIZE];
			long int sigma = (shift != NULL) ? shift[i] : 0;
			int comp;

			for(comp = 0; comp < TDSIZE; comp++)
			{
				org_row[comp] = tmp_org_mat->element[comp] + i * org_stride;
				high_td[comp] = 0.0;
			}

			for(j = 0; j < col_dim; j++)
			{
				// (x + s) - s keeps the leading bits of x; valid under the IEEE
				// semantics this library is compiled with (no -ffast-math)
				high_ij = ret_row[j] + s;
				high_ij = high_ij - s;
				ret_row[j] = high_ij;

				// low_mat := mat - 2^shift * high_mat
				for(comp = 0; comp < TDSIZE; comp++)
					org_ij[comp] = org_row[comp][j];
				high_td[0] = (sigma != 0) ? bnc_oz_ldexp(high_ij, sigma) : high_ij;

				rtd_sub(rest_ij, org_ij, high_td);

				for(comp = 0; comp < TDSIZE; comp++)
					org_row[comp][j] = rest_ij[comp];
			}
		}

		real_num_div = index + 1;
	}

	free(power2);
	free_tdmatrix(tmp_org_mat);
	if(own_ret_mat != NULL)
		free_dmatrix(own_ret_mat);

	return real_num_div;
}

// SplitMat_A without the scaling; kept for callers that cannot apply a scale factor
int split_tdmatrix_dmat(DMatrix ret_mat[], int num_div, TDMatrix org_mat)
{
	return split_tdmatrix_dmat_ex(ret_mat, NULL, num_div, org_mat);
}

// absmax_col_tdmatrix
void absmax_col_tdmatrix(double mu[TDSIZE], long int *max_i, long int col_index, TDMatrix mat)
{
    long int i, max_index = 0;
    double abs_aij[TDSIZE];

	//mu = fabs(mat[0 * col_dim + j]);
    rtd_abs(mu, get_tdmatrix_ij(mat, 0, col_index));
	for(i = 1; i < mat->row_dim; i++)
	{
		rtd_abs(abs_aij, get_tdmatrix_ij(mat, i, col_index));
		if(rtd_cmp(abs_aij, mu) > 0)
        {
			rtd_set(mu, abs_aij);
            max_index = i;
        }
	}

    if(max_i != NULL)
        *max_i = max_index;

    return;
}

// SplitMat_B
/*------------------------------------------------------------------------------*/
/* SplitMat_B: same as split_tdmatrix_dmat, but the threshold of the second   */
/* operand is taken over columns, so the maxima are reduced per column.  The     */
/* sweep still walks the matrix row-wise -- one partial maximum vector per       */
/* thread, combined afterwards -- because the column-wise walk the definition    */
/* suggests would touch a new cache line on every element.                       */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
/* SplitMat_B: same as split_tdmatrix_dmat_ex(), but the threshold and the    */
/* scale of the second operand are taken over columns, so the maxima are reduced */
/* per column -- one partial maximum vector per thread, combined afterwards, so  */
/* that the sweep can still walk the matrix row-wise.                            */
/*------------------------------------------------------------------------------*/
int split_tdmatrix_t_dmat_ex(DMatrix ret_mat[], long int col_shift[], int num_div, TDMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, org_stride, ret_stride;
	int real_num_div, num_digits = 53, num_threads, thread; // IEEE double prec.
	double mu_total, tail_exp;
	TDMatrix tmp_org_mat;
	DMatrix own_ret_mat = NULL, in_ret_mat;
	double *power2, *mu_local; // 2^t_exp of each column, per-thread column maxima
	long int *shift;

	row_dim = org_mat->row_dim;
	col_dim = org_mat->col_dim;

	if(ret_mat != NULL)
	{
		if((ret_mat[0]->row_dim != row_dim) || (ret_mat[0]->col_dim != col_dim))
		{
			fprintf(stderr, "ERROR: split_tdmatrix_t_dmat\n");
			return 0;
		}
	}

	num_threads = bnc_oz_get_num_threads();

	power2 = (double *)calloc((size_t)col_dim, sizeof(double));
	mu_local = (double *)calloc((size_t)num_threads * (size_t)col_dim, sizeof(double));
	if(mu_local == NULL) // retry single-threaded rather than give up
	{
		num_threads = 1;
		mu_local = (double *)calloc((size_t)col_dim, sizeof(double));
	}
	if(power2 == NULL || mu_local == NULL)
	{
		fprintf(stderr, "ERROR: split_tdmatrix_t_dmat: cannot allocate\n");
		free(power2);
		free(mu_local);
		return 0;
	}

	if(ret_mat == NULL)
	{
		own_ret_mat = init_dmatrix(row_dim, col_dim);
		in_ret_mat = own_ret_mat;
	}
	else
		in_ret_mat = ret_mat[0];

	tmp_org_mat = init_tdmatrix(row_dim, col_dim);
	subst_tdmatrix(tmp_org_mat, org_mat);
	org_stride = tmp_org_mat->real_col_dim;

	// the column-independent half of t_exp
	tail_exp = ceil(((double)num_digits + DLOG2((double)(row_dim))) / 2.0);

	real_num_div = 0;
	for(index = 0; index < num_div; index++)
	{
		if(ret_mat != NULL)
			in_ret_mat = ret_mat[index];
		ret_stride = in_ret_mat->real_col_dim;
		shift = (col_shift != NULL) ? (col_shift + (size_t)index * (size_t)col_dim) : NULL;

		for(i = 0; i < (long int)num_threads * col_dim; i++)
			mu_local[i] = 0.0;

		// pass 1: the column maxima of what is left to split
#ifdef _OPENMP
		#pragma omp parallel num_threads(num_threads) private(i, j)
#endif // _OPENMP
		{
			double *local_mu;
#ifdef _OPENMP
			local_mu = mu_local + (size_t)omp_get_thread_num() * (size_t)col_dim;
			#pragma omp for schedule(static)
#else // _OPENMP
			local_mu = mu_local;
#endif // _OPENMP
			for(i = 0; i < row_dim; i++)
			{
				const double *org_row = tmp_org_mat->element[0] + i * org_stride;
				double abs_org_ij;

				for(j = 0; j < col_dim; j++)
				{
					abs_org_ij = fabs(org_row[j]);
					if(abs_org_ij > local_mu[j])
						local_mu[j] = abs_org_ij;
				}
			}
		}

		// combine the per-thread maxima, pick the column scale and the threshold
		mu_total = 0.0;
		for(j = 0; j < col_dim; j++)
		{
			double mu = mu_local[j];
			long int sigma = 0;

			for(thread = 1; thread < num_threads; thread++)
			{
				if(mu_local[(size_t)thread * (size_t)col_dim + j] > mu)
					mu = mu_local[(size_t)thread * (size_t)col_dim + j];
			}

			if(shift != NULL)
			{
				sigma = bnc_oz_exp2_d(mu);
				if(sigma < BNC_OZ_MIN_SCALED_EXP)
					sigma = 0;
				shift[j] = sigma;

				if(sigma != 0)
					mu = bnc_oz_ldexp(mu, -sigma);
			}

			power2[j] = pow(2.0, ceil(DLOG2(mu)) + tail_exp);
			mu_total += mu;
		}

		if(mu_total == 0.0) break;

		// pass 2: scale, high := (mat + s) - s, and mat := mat - 2^shift[j] * high
#ifdef _OPENMP
		#pragma omp parallel for num_threads(num_threads) schedule(static) private(i, j)
#endif // _OPENMP
		for(i = 0; i < row_dim; i++)
		{
			double *ret_row = in_ret_mat->element + i * ret_stride;
			double *org_row[TDSIZE];
			double s, high_ij, scaled_ij;
			double org_ij[TDSIZE], high_td[TDSIZE], rest_ij[TDSIZE];
			long int sigma;
			int comp;

			for(comp = 0; comp < TDSIZE; comp++)
			{
				org_row[comp] = tmp_org_mat->element[comp] + i * org_stride;
				high_td[comp] = 0.0;
			}

			for(j = 0; j < col_dim; j++)
			{
				sigma = (shift != NULL) ? shift[j] : 0;
				s = power2[j];

				scaled_ij = (sigma != 0) ? bnc_oz_ldexp(org_row[0][j], -sigma) : org_row[0][j];

				high_ij = scaled_ij + s;
				high_ij = high_ij - s;
				ret_row[j] = high_ij;

				for(comp = 0; comp < TDSIZE; comp++)
					org_ij[comp] = org_row[comp][j];
				high_td[0] = (sigma != 0) ? bnc_oz_ldexp(high_ij, sigma) : high_ij;

				rtd_sub(rest_ij, org_ij, high_td);

				for(comp = 0; comp < TDSIZE; comp++)
					org_row[comp][j] = rest_ij[comp];
			}
		}

		real_num_div = index + 1;
	}

	free(power2);
	free(mu_local);
	free_tdmatrix(tmp_org_mat);
	if(own_ret_mat != NULL)
		free_dmatrix(own_ret_mat);

	return real_num_div;
}

// SplitMat_B without the scaling; kept for callers that cannot apply a scale factor
int split_tdmatrix_t_dmat(DMatrix ret_mat[], int num_div, TDMatrix org_mat)
{
	return split_tdmatrix_t_dmat_ex(ret_mat, NULL, num_div, org_mat);
}

// Matrix multiplication based on Ozaki scheme
/*------------------------------------------------------------------------------*/
/* Matrix multiplication based on Ozaki scheme                                   */
/*                                                                               */
/* ret = sum_{i,j} div_a[i] * div_b[j].  In BNC_OZ_GEMM_MODE_OWN the rows of ret */
/* are cut into blocks and one thread takes a block at a time, running every     */
/* slice product for it with a single-threaded DGEMM and accumulating in TD   */
/* on the spot; the accumulation, which a threaded BLAS would leave serial, is   */
/* thereby parallelized as well and the block of ret stays hot in cache across   */
/* all the slice pairs.  Blocks are disjoint and each element of ret still sums  */
/* its slice products in the original order, so the result is bit-identical to   */
/* the serial one.  BNC_OZ_GEMM_MODE_BLAS keeps the old shape (one full DGEMM    */
/* per slice pair, left to the BLAS to thread).                                  */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
/* Matrix multiplication based on Ozaki scheme                                   */
/*                                                                               */
/* ret = sum_{i,j} 2^(sa[i] + sb[j]) div_a[i] * div_b[j].  In                    */
/* BNC_OZ_GEMM_MODE_OWN the rows of ret are cut into blocks and one thread takes */
/* a block at a time, running every slice product for it with a single-threaded  */
/* DGEMM and accumulating in TD on the spot; the accumulation, which a        */
/* threaded BLAS would leave serial, is thereby parallelized as well and the     */
/* block of ret stays hot in cache across all the slice pairs.  Blocks are       */
/* disjoint and each element of ret still sums its slice products in the         */
/* original order, so the result does not depend on the number of threads.       */
/* BNC_OZ_GEMM_MODE_BLAS keeps the old shape (one full DGEMM per slice pair,     */
/* left to the BLAS to thread).                                                  */
/*------------------------------------------------------------------------------*/
void mul_tdmatrix_oz(TDMatrix ret, TDMatrix a, int max_num_div_a, TDMatrix b, int max_num_div_b)
{
    int i, j;
    int real_num_div_a, real_num_div_b, num_threads, prev_blas_threads;
    long int row_dim = ret->row_dim, col_dim = ret->col_dim, mid_dim = a->col_dim;
    long int block_rows, num_blocks;
    long int *row_shift, *col_shift;
    DMatrix *div_a, *div_b, div_ret;
    double *block_buf = NULL;

    if(mid_dim != b->row_dim)
    {
        fprintf(stderr, "ERROR: mul_tdmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
        return;
    }

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_b = (DMatrix *)calloc(max_num_div_b, sizeof(DMatrix));
    row_shift = (long int *)calloc((size_t)max_num_div_a * (size_t)row_dim, sizeof(long int));
    col_shift = (long int *)calloc((size_t)max_num_div_b * (size_t)col_dim, sizeof(long int));
    if(div_a == NULL || div_b == NULL || row_shift == NULL || col_shift == NULL)
    {
        fprintf(stderr, "ERROR: mul_tdmatrix_oz: cannot allocate\n");
        free(div_a); free(div_b); free(row_shift); free(col_shift);
        return;
    }
    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, mid_dim);
    for(i = 0; i < max_num_div_b; i++)
        div_b[i] = init_dmatrix(mid_dim, col_dim);

    real_num_div_a = split_tdmatrix_dmat_ex(div_a, row_shift, max_num_div_a, a);
    real_num_div_b = split_tdmatrix_t_dmat_ex(div_b, col_shift, max_num_div_b, b);

    set0_tdmatrix(ret);

    num_threads = bnc_oz_get_num_threads();
    block_rows = bnc_oz_block_rows_for(row_dim, num_threads);
    num_blocks = (row_dim + block_rows - 1) / block_rows;

    if((bnc_oz_get_gemm_mode() == BNC_OZ_GEMM_MODE_OWN) && (num_threads > 1))
        block_buf = (double *)malloc((size_t)num_threads * (size_t)block_rows * (size_t)col_dim * sizeof(double));

    if(block_buf != NULL)
    {
        prev_blas_threads = bnc_oz_blas_enter();

#ifdef _OPENMP
        #pragma omp parallel num_threads(num_threads)
#endif // _OPENMP
        {
            long int blk, row_start, num_rows, ii, jj, shift_a;
            int div_i, div_j, comp;
            double *buf;
            double *ret_row[TDSIZE];
            double ret_ij[TDSIZE], add_ij[TDSIZE], sum_ij[TDSIZE];

            for(comp = 0; comp < TDSIZE; comp++)
                add_ij[comp] = 0.0;

#ifdef _OPENMP
            buf = block_buf + (size_t)omp_get_thread_num() * (size_t)block_rows * (size_t)col_dim;
            #pragma omp for schedule(dynamic, 1)
#else // _OPENMP
            buf = block_buf;
#endif // _OPENMP
            for(blk = 0; blk < num_blocks; blk++)
            {
                row_start = blk * block_rows;
                num_rows = ((row_dim - row_start) < block_rows) ? (row_dim - row_start) : block_rows;

                for(div_i = 0; div_i < real_num_div_a; div_i++)
                {
                    for(div_j = 0; div_j < real_num_div_b - div_i; div_j++)
                    {
                        const long int *shift_b = col_shift + (size_t)div_j * (size_t)col_dim;

                        bnc_oz_dgemm_block(buf, col_dim, div_a[div_i], row_start, num_rows, div_b[div_j]);

                        for(ii = 0; ii < num_rows; ii++)
                        {
                            const double *buf_row = buf + ii * col_dim;

                            shift_a = row_shift[(size_t)div_i * (size_t)row_dim + row_start + ii];

                            for(comp = 0; comp < TDSIZE; comp++)
                                ret_row[comp] = ret->element[comp] + (row_start + ii) * ret->real_col_dim;

                            for(jj = 0; jj < col_dim; jj++)
                            {
                                for(comp = 0; comp < TDSIZE; comp++)
                                    ret_ij[comp] = ret_row[comp][jj];
                                add_ij[0] = bnc_oz_ldexp(buf_row[jj], shift_a + shift_b[jj]);

                                rtd_add(sum_ij, ret_ij, add_ij);

                                for(comp = 0; comp < TDSIZE; comp++)
                                    ret_row[comp][jj] = sum_ij[comp];
                            }
                        }
                    }
                }
            }
        }

        bnc_oz_blas_leave(prev_blas_threads);
        free(block_buf);
    }
    else // one full-size product per slice pair, threaded by the BLAS if it can
    {
        long int ii, jj;
        int comp;
        double *ret_row[TDSIZE];
        double ret_ij[TDSIZE], add_ij[TDSIZE], sum_ij[TDSIZE];

        for(comp = 0; comp < TDSIZE; comp++)
            add_ij[comp] = 0.0;

        div_ret = init_dmatrix(row_dim, col_dim);

        for(i = 0; i < real_num_div_a; i++)
        {
            for(j = 0; j < real_num_div_b - i; j++)
            {
                const long int *shift_b = col_shift + (size_t)j * (size_t)col_dim;

                bnc_oz_dgemm_block(div_ret->element, div_ret->real_col_dim, div_a[i], 0, row_dim, div_b[j]);

                for(ii = 0; ii < row_dim; ii++)
                {
                    const double *buf_row = div_ret->element + ii * div_ret->real_col_dim;
                    long int shift_a = row_shift[(size_t)i * (size_t)row_dim + ii];

                    for(comp = 0; comp < TDSIZE; comp++)
                        ret_row[comp] = ret->element[comp] + ii * ret->real_col_dim;

                    for(jj = 0; jj < col_dim; jj++)
                    {
                        for(comp = 0; comp < TDSIZE; comp++)
                            ret_ij[comp] = ret_row[comp][jj];
                        add_ij[0] = bnc_oz_ldexp(buf_row[jj], shift_a + shift_b[jj]);

                        rtd_add(sum_ij, ret_ij, add_ij);

                        for(comp = 0; comp < TDSIZE; comp++)
                            ret_row[comp][jj] = sum_ij[comp];
                    }
                }
            }
        }

        free_dmatrix(div_ret);
    }

    for(i = 0; i < max_num_div_a; i++)
        free_dmatrix(div_a[i]);
    for(i = 0; i < max_num_div_b; i++)
        free_dmatrix(div_b[i]);

    free(div_a);
    free(div_b);
    free(row_shift);
    free(col_shift);
}

// Matrix-Vector multiplication based on Ozaki scheme
/*------------------------------------------------------------------------------*/
/* Matrix-Vector multiplication based on Ozaki scheme                            */
/*                                                                               */
/* Same blocking as mul_tdmatrix_oz: a thread owns a range of rows of ret and */
/* runs all slice pairs for it, so the accumulation is parallel too.             */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
/* Matrix-Vector multiplication based on Ozaki scheme                            */
/*                                                                               */
/* Same blocking and the same exponent handling as mul_tdmatrix_oz(): a       */
/* thread owns a range of rows of ret and runs all slice pairs for it.           */
/*------------------------------------------------------------------------------*/
void mul_tdmatrix_tdvec_oz(TDVector ret, TDMatrix a, int max_num_div_a, TDVector vb, int max_num_div_vb)
{
    int i, j;
    int real_num_div_a, real_num_div_vb, num_threads, prev_blas_threads;
    long int vec_dim = ret->dim, row_dim = a->row_dim, col_dim = a->col_dim;
    long int block_rows, num_blocks;
    long int *row_shift, *vec_shift;
    DMatrix *div_a;
    DVector *div_vb, div_ret;
    double *block_buf = NULL;

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_vb = (DVector *)calloc(max_num_div_vb, sizeof(DVector));
    row_shift = (long int *)calloc((size_t)max_num_div_a * (size_t)row_dim, sizeof(long int));
    vec_shift = (long int *)calloc((size_t)max_num_div_vb, sizeof(long int));
    if(div_a == NULL || div_vb == NULL || row_shift == NULL || vec_shift == NULL)
    {
        fprintf(stderr, "ERROR: mul_tdmatrix_tdvec_oz: cannot allocate\n");
        free(div_a); free(div_vb); free(row_shift); free(vec_shift);
        return;
    }

    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, col_dim);
    for(i = 0; i < max_num_div_vb; i++)
        div_vb[i] = init_dvector(vb->dim); // vb->dim, not ret->dim: they differ when a is not square

    real_num_div_a = split_tdmatrix_dmat_ex(div_a, row_shift, max_num_div_a, a);
    real_num_div_vb = split_tdvector_dvec_ex(div_vb, vec_shift, max_num_div_vb, vb);

    set0_tdvector(ret);

    num_threads = bnc_oz_get_num_threads();
    block_rows = bnc_oz_block_rows_for(vec_dim, num_threads);
    num_blocks = (vec_dim + block_rows - 1) / block_rows;

    if((bnc_oz_get_gemm_mode() == BNC_OZ_GEMM_MODE_OWN) && (num_threads > 1))
        block_buf = (double *)malloc((size_t)num_threads * (size_t)block_rows * sizeof(double));

    if(block_buf != NULL)
    {
        prev_blas_threads = bnc_oz_blas_enter();

#ifdef _OPENMP
        #pragma omp parallel num_threads(num_threads)
#endif // _OPENMP
        {
            long int blk, row_start, num_rows, ii, shift_a;
            int div_i, div_j, comp;
            double *buf;
            double ret_i[TDSIZE], add_i[TDSIZE], sum_i[TDSIZE];

            for(comp = 0; comp < TDSIZE; comp++)
                add_i[comp] = 0.0;

#ifdef _OPENMP
            buf = block_buf + (size_t)omp_get_thread_num() * (size_t)block_rows;
            #pragma omp for schedule(dynamic, 1)
#else // _OPENMP
            buf = block_buf;
#endif // _OPENMP
            for(blk = 0; blk < num_blocks; blk++)
            {
                row_start = blk * block_rows;
                num_rows = ((vec_dim - row_start) < block_rows) ? (vec_dim - row_start) : block_rows;

                for(div_i = 0; div_i < real_num_div_a; div_i++)
                {
                    for(div_j = 0; div_j < real_num_div_vb; div_j++)
                    {
                        bnc_oz_dgemv_block(buf, div_a[div_i], row_start, num_rows, div_vb[div_j]);

                        for(ii = 0; ii < num_rows; ii++)
                        {
                            shift_a = row_shift[(size_t)div_i * (size_t)row_dim + row_start + ii];

                            for(comp = 0; comp < TDSIZE; comp++)
                                ret_i[comp] = ret->element[comp][row_start + ii];
                            add_i[0] = bnc_oz_ldexp(buf[ii], shift_a + vec_shift[div_j]);

                            rtd_add(sum_i, ret_i, add_i);

                            for(comp = 0; comp < TDSIZE; comp++)
                                ret->element[comp][row_start + ii] = sum_i[comp];
                        }
                    }
                }
            }
        }

        bnc_oz_blas_leave(prev_blas_threads);
        free(block_buf);
    }
    else
    {
        long int ii;
        int comp;
        double ret_i[TDSIZE], add_i[TDSIZE], sum_i[TDSIZE];

        for(comp = 0; comp < TDSIZE; comp++)
            add_i[comp] = 0.0;

        div_ret = init_dvector(vec_dim);

        for(i = 0; i < real_num_div_a; i++)
        {
            for(j = 0; j < real_num_div_vb; j++)
            {
                bnc_oz_dgemv_block(div_ret->element, div_a[i], 0, vec_dim, div_vb[j]);

                for(ii = 0; ii < vec_dim; ii++)
                {
                    long int shift_a = row_shift[(size_t)i * (size_t)row_dim + ii];

                    for(comp = 0; comp < TDSIZE; comp++)
                        ret_i[comp] = ret->element[comp][ii];
                    add_i[0] = bnc_oz_ldexp(div_ret->element[ii], shift_a + vec_shift[j]);

                    rtd_add(sum_i, ret_i, add_i);

                    for(comp = 0; comp < TDSIZE; comp++)
                        ret->element[comp][ii] = sum_i[comp];
                }
            }
        }

        free_dvector(div_ret);
    }

    for(i = 0; i < max_num_div_a; i++)
        free_dmatrix(div_a[i]);
    for(i = 0; i < max_num_div_vb; i++)
        free_dvector(div_vb[i]);

    free(div_a);
    free(div_vb);
    free(row_shift);
    free(vec_shift);
}

// Fit dimension to be multiple of min_dim
void mul_ctdmatrix_oz_3m(CTDMatrix ret, CTDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CTDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    TDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tdmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_tdmatrix(b->re->row_dim, ret->re->col_dim);

    mul_tdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    mul_tdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_b_image);
    sub_tdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_tdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_tdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        add_tdmatrix(ret->im, t1, t2);
    #else // USE_4M
    */
        // 3M
        add_tdmatrix(t3, a->re, a->im);
        add_tdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_tdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_tdmatrix(ret->im, ret->im, t1);
        sub_tdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_tdmatrix(t1);
    free_tdmatrix(t2);
    free_tdmatrix(t3);
    free_tdmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void mul_ctdmatrix_oz_4m(CTDMatrix ret, CTDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CTDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    TDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_tdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    mul_tdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_b_image);
    sub_tdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_tdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_tdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        add_tdmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_tdmatrix(t3, a->re, a->im);
        add_tdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_tdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_tdmatrix(ret->im, ret->im, t1);
        sub_tdmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_tdmatrix(t1);
    free_tdmatrix(t2);
    free_tdmatrix(t3);
    free_tdmatrix(t4);
}


// Testing
#ifdef DEBUG

//#define DIM 5
//#define DIM 10
#define DIM 128
//#define DIM 512

//#define MAX_NUM_DIV 10
#define MAX_NUM_DIV 15

int main()
{
    long int i, j;
    int real_num_div;
    double tmp[TDSIZE], tmp2[TDSIZE];
    TDVector vec_org, vec;
    DVector dvec[MAX_NUM_DIV];
    TDMatrix mat_a, mat_b, mat_c, mat;
    DMatrix dmat[MAX_NUM_DIV];
    tdfloat ddtmp, ddtmp2;

    //goto tdmatrix;

// DVector 
    vec_org = init_tdvector(DIM);
    vec = init_tdvector(DIM);
    for(i = 0; i < MAX_NUM_DIV; i++)
        dvec[i] = init_dvector(DIM);

    // set random
    srand(DIM);
    for(i = 0; i < DIM; i++)
    {
        rtd_set_d(tmp, (double)rand());
        rtd_mul_d(tmp, tmp, (double)rand());
        rtd_div_d(tmp, tmp, (double)rand());
        set_tdvector_i(vec_org, i, tmp);
    }

    // split
    real_num_div = split_tdvector_dvec(dvec, MAX_NUM_DIV, vec_org); //, 53);

    for(i = 0; i < MAX_NUM_DIV; i++)
    {
        //printf("dvec[%d]:\n", i); print_dvector(dvec[i]);

        add_tdvector_dvec(vec, vec, dvec[i]);
    }
    //sub_tdvector(vec, vec, vec_org);
    //printf("vec - vec_org:\n"); print_tdvector(vec);
    //printf("\n");
    for(i = 0; i < DIM; i++)
    {
        ddtmp = get_tdvector_i_tdfloat(vec, i);
        ddtmp2 = get_tdvector_i_tdfloat(vec_org, i);
        printf("%5d %25.17e %25.17e %25.17e\n%5d %25.17e %25.17e %25.17e\n", i, ddtmp.val[0], ddtmp.val[1], ddtmp.val[2], i, ddtmp2.val[0], ddtmp2.val[1], ddtmp.val[2]);
    }
    

    free_tdvector(vec_org);
    free_tdvector(vec);
    for(i = 0; i < 10; i++)
        free_dvector(dvec[i]);

    //sgoto end;
// DMatrix
tdmatrix:

    mat_a = init_tdmatrix(DIM, DIM);
    mat_b = init_tdmatrix(DIM, DIM);
    mat_c = init_tdmatrix(DIM, DIM);
    mat = init_tdmatrix(DIM, DIM);
    for(i = 0; i < MAX_NUM_DIV; i++)
        dmat[i] = init_dmatrix(DIM, DIM);

    // set random
    srand(DIM);
    for(i = 0; i < DIM; i++)
    {
        for(j = 0; j < DIM; j++)
        {
            rtd_set_d(tmp, (double)rand() / (double)RAND_MAX);
            rtd_mul_d(tmp, tmp, (double)rand());
            rtd_div_d(tmp, tmp, (double)rand());
            set_tdmatrix_ij(mat_a, i, j, tmp);

            rtd_set_d(tmp, (double)rand() / (double)RAND_MAX);
            rtd_mul_d(tmp, tmp, (double)rand());
            rtd_div_d(tmp, tmp, (double)rand());
            set_tdmatrix_ij(mat_b, i, j, tmp);
        }
    }

    // split_A
    printf("Split_A:\n");
    real_num_div = split_tdmatrix_dmat(dmat, MAX_NUM_DIV, mat_a);
    printf("num_div, real_num_div = %d, %d\n", MAX_NUM_DIV, real_num_div);

    for(i = 0; i < real_num_div; i++)
    {
        //printf("dmat[%d]:\n", i); print_dmatrix(dmat[i]);

        add_tdmatrix_dmat(mat, mat, dmat[i]);
    }
    sub_tdmatrix(mat, mat, mat_a);
    printf("||mat - mat_org|| / ||mat_org||:\n"); 
    normf_tdmatrix(tmp, mat);
    normf_tdmatrix(tmp2, mat_a);
    rtd_div(tmp, tmp, tmp2);
    rtd_out_str(tmp); //print_tdmatrix(mat);
    printf("\n"); 

    // split_B
    printf("Split_B:\n");
    real_num_div = split_tdmatrix_t_dmat(dmat, MAX_NUM_DIV, mat_b);
    printf("num_div, real_num_div = %d, %d\n", MAX_NUM_DIV, real_num_div);

    for(i = 0; i < real_num_div; i++)
    {
        //printf("dmat[%d]:\n", i); print_dmatrix(dmat[i]);

        add_tdmatrix_dmat(mat, mat, dmat[i]);
    }
    sub_tdmatrix(mat, mat, mat_b);
    printf("||mat - mat_org|| / ||mat_org||:\n"); 
    normf_tdmatrix(tmp, mat);
    normf_tdmatrix(tmp2, mat_a);
    rtd_div(tmp, tmp, tmp2);
    rtd_out_str(tmp); //print_tdmatrix(mat);
    printf("\n"); 

    // a * b
    mul_tdmatrix_oz(mat, mat_a, MAX_NUM_DIV, mat_b, MAX_NUM_DIV);
    mul_tdmatrix(mat_c, mat_a, mat_b);

    for(i = 0; i < mat->row_dim; i++)
    {
        for(j = 0; j < mat->col_dim; j++)
        {
            ddtmp = get_tdmatrix_ij_tdfloat(mat, i, j);
            ddtmp2 = get_tdmatrix_ij_tdfloat(mat_c, i, j);
            if(ddtmp.val[1] != ddtmp2.val[1])
                printf("%5d, %5d %25.17e %25.17e %25.17e\n%5d, %5d %25.17e %25.17e %25.17e\n", i, j, ddtmp.val[0], ddtmp.val[1], ddtmp.val[2], i, j, ddtmp2.val[0], ddtmp2.val[1], ddtmp2.val[2]);
        }
    }

    sub_tdmatrix(mat, mat, mat_c);
    printf("||mat_oz - mat_org|| / ||mat_org||:\n"); 
    normf_tdmatrix(tmp, mat);
    normf_tdmatrix(tmp2, mat_c);
    rtd_div(tmp, tmp, tmp2);
    rtd_out_str(tmp); //print_tdmatrix(mat); 
    printf("\n"); 

    free_tdmatrix(mat_a);
    free_tdmatrix(mat_b);
    free_tdmatrix(mat_c);
    free_tdmatrix(mat);
    for(i = 0; i < 10; i++)
        free_dmatrix(dmat[i]);

end:
    return 0;
}

// Testing
#endif // DEBUG
