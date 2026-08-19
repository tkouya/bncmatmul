/********************************************************************************/
/* dd_oz_scheme: Multiple precision linear computation based on Ozaki scheme.   */
/* Copyright (C) 2022-2024 Tomonori Kouya                                       */
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

// log2(x) := log10(x) / log10(2)
//#define DLOG2(x) (log10((x)) / 0.30102999566398119521373889472449)

// absmax_ddvector
void absmax_ddvector(double ret[DDSIZE], long int *max_index, DDVector vec)
{
    long int i, max_i, dim = vec->dim;
    double abs_val[DDSIZE];

    max_i = 0;
    ret[0] = 0.0; ret[1] = 0.0;
    for(i = 0; i < dim; i++)
    {
        //abs_val = fabs(get_ddvector_i(vec, i));
        rdd_abs(abs_val, get_ddvector_i(vec, i));
        if(rdd_cmp(ret, abs_val) < 0)
        {
            rdd_set(ret, abs_val);
            max_i = i;
        }
    }

    if(max_index != NULL)
        *max_index = max_i;

    return;
}

/* c = a + (double)b */
void add_ddvector_dvec(DDVector c, DDVector a, DVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_ddvector_dvec\n");
		return;
	}

	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_add_d(tmp, get_ddvector_i(a, i), get_dvector_i(b, i));
		set_ddvector_i(c, i, tmp);
	}
}

/* c = a - (double)b */
void sub_ddvector_dvec(DDVector c, DDVector a, DVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_ddvector_dvec\n");
		return;
	}

	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_sub_d(tmp, get_ddvector_i(a, i),  get_dvector_i(b, i));
		set_ddvector_i(c, i, tmp);
	}
}

// split vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
//void extract_ddvector(DVector ret_vec[], int num_div, DDVector org_vec, int num_bits)
/*------------------------------------------------------------------------------*/
/* split vector: 2^shift[index] * (ret_vec[0] + ... ) = org_vec                  */
/*                                                                               */
/* The vector plays the role of a single column of B, so it carries one exponent */
/* per slice.  shift may be NULL; see oz_scheme.h.                               */
/*------------------------------------------------------------------------------*/
int split_ddvector_dvec_ex(DVector ret_vec[], long int shift[], int num_div, DDVector org_vec)
{
	long int dim = org_vec->dim, i;
	int index, num_bits = 53, real_num_div; // IEEE754 binary64
	double absmax_org_vec, threshold, t_exp, tail_exp, org_vec_i, high_i;
	double org_ii[DDSIZE], high_dd[DDSIZE], rest_i[DDSIZE];
	long int sigma;
	DDVector tmp_org_vec;
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
	tmp_org_vec = init_ddvector(dim);
	subst_ddvector(tmp_org_vec, org_vec);

	tail_exp = ceil(((double)num_bits + DLOG2((double)(dim))) / 2.0);

	for(comp = 0; comp < DDSIZE; comp++)
		high_dd[comp] = 0.0;

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

			for(comp = 0; comp < DDSIZE; comp++)
				org_ii[comp] = tmp_org_vec->element[comp][i];
			high_dd[0] = (sigma != 0) ? bnc_oz_ldexp(high_i, sigma) : high_i;

			rdd_sub(rest_i, org_ii, high_dd);

			for(comp = 0; comp < DDSIZE; comp++)
				tmp_org_vec->element[comp][i] = rest_i[comp];
		}

		real_num_div = index + 1;
	}

	free_ddvector(tmp_org_vec);
	if(own_ret_vec != NULL)
		free_dvector(own_ret_vec);

	return real_num_div;
}

// split vector without the scaling
int split_ddvector_dvec(DVector ret_vec[], int num_div, DDVector org_vec)
{
	return split_ddvector_dvec_ex(ret_vec, NULL, num_div, org_vec);
}

// absmax_row_ddmatrix
void absmax_row_ddmatrix(double mu[DDSIZE], long int *max_j, long int row_index, DDMatrix mat)
{
    long int j, max_index = 0;
    double abs_aij[DDSIZE];

	//mu = fabs(mat[i * col_dim + 0]);
    rdd_abs(mu, get_ddmatrix_ij(mat, row_index, 0));

	for(j = 1; j < mat->real_col_dim; j++)
	{
		rdd_abs(abs_aij, get_ddmatrix_ij(mat, row_index, j));
		if(rdd_cmp(abs_aij, mu) > 0)
        {
			//mu = abs_aij;
            rdd_set(mu, abs_aij);
            max_index = j;
        }
	}

    if(max_j != NULL)
        *max_j = max_index;

    //return mu;
    return;
}

/* c := a + (double)b */
void add_ddmatrix_dmat(DDMatrix c, DDMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim, a_stride, b_stride, c_stride;
	int num_threads;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_ddmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_ddmatrix_dmat\n");
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
		const double *a_hi = a->element[0] + i * a_stride;
		const double *a_lo = a->element[1] + i * a_stride;
		const double *b_row = b->element + i * b_stride;
		double *c_hi = c->element[0] + i * c_stride;
		double *c_lo = c->element[1] + i * c_stride;
		double a_ij[DDSIZE], b_ij[DDSIZE], c_ij[DDSIZE];

		for(j = 0; j < col_dim; j++)
		{
			a_ij[0] = a_hi[j]; a_ij[1] = a_lo[j];
			b_ij[0] = b_row[j]; b_ij[1] = 0.0;
			rdd_add(c_ij, a_ij, b_ij);
			c_hi[j] = c_ij[0]; c_lo[j] = c_ij[1];
		}
	}
}

/* c := a - (double)b */
void sub_ddmatrix_dmat(DDMatrix c, DDMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim, a_stride, b_stride, c_stride;
	int num_threads;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_ddmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_ddmatrix_dmat\n");
		return;
	}
	col_dim = c->col_dim;

	a_stride = a->real_col_dim;
	b_stride = b->real_col_dim;
	c_stride = c->real_col_dim;

	num_threads = bnc_oz_get_num_threads();

#ifdef _OPENMP
	#pragma omp parallel for num_threads(num_threads) schedule(static) private(i, j)
#endif // _OPENMP
	for(i = 0; i < row_dim; i++)
	{
		const double *a_hi = a->element[0] + i * a_stride;
		const double *a_lo = a->element[1] + i * a_stride;
		const double *b_row = b->element + i * b_stride;
		double *c_hi = c->element[0] + i * c_stride;
		double *c_lo = c->element[1] + i * c_stride;
		double a_ij[DDSIZE], b_ij[DDSIZE], c_ij[DDSIZE];

		for(j = 0; j < col_dim; j++)
		{
			a_ij[0] = a_hi[j]; a_ij[1] = a_lo[j];
			b_ij[0] = b_row[j]; b_ij[1] = 0.0;
			rdd_sub(c_ij, a_ij, b_ij);
			c_hi[j] = c_ij[0]; c_lo[j] = c_ij[1];
		}
	}
}

//#define SPLIT_NUM_DIGITS 53
//#define SPLIT_NUM_DIGITS 55
//#define SPLIT_NUM_DIGITS 56
//#define SPLIT_NUM_DIGITS 57
//#define SPLIT_NUM_DIGITS 64

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
// return real_num_div
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
int split_ddmatrix_dmat_ex(DMatrix ret_mat[], long int row_shift[], int num_div, DDMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, org_stride, ret_stride;
	long int num_digits = 53; // IEEE double prec.
	int real_num_div, num_threads;
	double mu_total, tail_exp;
	DDMatrix tmp_org_mat;
	DMatrix own_ret_mat = NULL, in_ret_mat;
	double *power2; // 2^t_exp of each row, in the scaled domain
	long int *shift;

	row_dim = org_mat->row_dim;
	col_dim = org_mat->col_dim;

	if(ret_mat != NULL)
	{
		if((ret_mat[0]->row_dim != row_dim) || (ret_mat[0]->col_dim != col_dim))
		{
			fprintf(stderr, "ERROR: split_ddmatrix_dmat\n");
			return 0;
		}
	}

	power2 = (double *)calloc((size_t)row_dim, sizeof(double));
	if(power2 == NULL)
	{
		fprintf(stderr, "ERROR: split_ddmatrix_dmat: cannot allocate\n");
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
	tmp_org_mat = init_ddmatrix(row_dim, col_dim);
	subst_ddmatrix(tmp_org_mat, org_mat);
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
				abs_org_ij = fabs(org_row[j]); // == |rdd_get_d(org[i][j])|
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
			double *org_row[DDSIZE];
			double s = power2[i], high_ij;
			double org_ij[DDSIZE], high_dd[DDSIZE], rest_ij[DDSIZE];
			long int sigma = (shift != NULL) ? shift[i] : 0;
			int comp;

			for(comp = 0; comp < DDSIZE; comp++)
			{
				org_row[comp] = tmp_org_mat->element[comp] + i * org_stride;
				high_dd[comp] = 0.0;
			}

			for(j = 0; j < col_dim; j++)
			{
				// (x + s) - s keeps the leading bits of x; valid under the IEEE
				// semantics this library is compiled with (no -ffast-math)
				high_ij = ret_row[j] + s;
				high_ij = high_ij - s;
				ret_row[j] = high_ij;

				// low_mat := mat - 2^shift * high_mat
				for(comp = 0; comp < DDSIZE; comp++)
					org_ij[comp] = org_row[comp][j];
				high_dd[0] = (sigma != 0) ? bnc_oz_ldexp(high_ij, sigma) : high_ij;

				rdd_sub(rest_ij, org_ij, high_dd);

				for(comp = 0; comp < DDSIZE; comp++)
					org_row[comp][j] = rest_ij[comp];
			}
		}

		real_num_div = index + 1;
	}

	free(power2);
	free_ddmatrix(tmp_org_mat);
	if(own_ret_mat != NULL)
		free_dmatrix(own_ret_mat);

	return real_num_div;
}

// SplitMat_A without the scaling; kept for callers that cannot apply a scale factor
int split_ddmatrix_dmat(DMatrix ret_mat[], int num_div, DDMatrix org_mat)
{
	return split_ddmatrix_dmat_ex(ret_mat, NULL, num_div, org_mat);
}

// split a sparse matrix to double sparse matrices
//#include "split_ddrsmatrix_drsmat.c"

// SplitMat_A
void split_ddmatrix2(DMatrix ret_mat[], int num_div, DDMatrix org_mat, int num_digits)
{
	long int i, j, index, row_dim, col_dim;
	//long int num_digits = 53; // IEEE double prec.
	//long int num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//long int num_digits = 64; // IEEE double prec.
	//double *s;
    DDMatrix tmp_org_mat;
    DDMatrix s, tmp_mat[2]; 
	double mu[DDSIZE], abs_aij[DDSIZE], t_exp[DDSIZE], power2[DDSIZE], two[DDSIZE] = {2.0, 0.0};

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;

	// threshold matrix 
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    s = init_ddmatrix(row_dim, col_dim);
    //s = init_dmatrix(row_dim, col_dim);

    tmp_mat[0] = init_ddmatrix(row_dim, col_dim);
    tmp_mat[1] = init_ddmatrix(row_dim, col_dim);
    //tmp_mat[0] = init_dmatrix(row_dim, col_dim);
    //tmp_mat[1] = init_dmatrix(row_dim, col_dim);

    // tmp_org_mat := org_mat;
    tmp_org_mat = init_ddmatrix(row_dim, col_dim);
    subst_ddmatrix(tmp_org_mat, org_mat);

    for(index = 0; index < num_div; index++)
    {
        //printf("In split_ddmatrix ... index= %d\n", index);
        //set0_dmatrix(ret_mat[index]);
        //subst_dmatrix_ddmat(ret_mat[index], tmp_org_mat);

        // mu[i] = max_j |mat[i, j]|
        for(i = 0; i < row_dim; i++)
        {
            absmax_row_ddmatrix(mu, NULL, i, tmp_org_mat);
            //mu = absmax_row_dmatrix(NULL, i, ret_mat[index]);

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);
            t_exp[1] = 0.0;
            //t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);

            // s[i, j] = 2^t_exp
            rdd_pow(power2, two, t_exp);
            rdd_pow_mpfr(power2, two, t_exp);
            //power2 = pow(2.0, t_exp);
            //printf("index, i, power2 = %ld, %ld, %25.17e\n", index, i, power2[0]);
            for(j = 0; j < col_dim; j++)
            {
                //s[i * col_dim + j] = pow(2.0, t_exp);
                set_ddmatrix_ij(s, i, j, power2);
                //set_dmatrix_ij(s, i, j, power2);
            }
        }

        // split org_mat to ret_high_mat and ret_low_mat

        // tmp_mat := mat + s
        add_ddmatrix(tmp_mat[0], tmp_org_mat, s);
        //add_dmatrix(tmp_mat[0], ret_mat[index], s);

        // high_mat := tmp_mat - s
        sub_ddmatrix(tmp_mat[1], tmp_mat[0], s);
        //sub_dmatrix(tmp_mat[1], tmp_mat[0], s);
        subst_dmatrix_ddmat(ret_mat[index], tmp_mat[1]);


        // low_mat := mat - high_mat
        sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
        //sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);
    }

	// free s
	free_ddmatrix(s);
	//free_dmatrix(s);
    free_ddmatrix(tmp_org_mat);
    //free_dmatrix(tmp_mat[0]);
    //free_dmatrix(tmp_mat[1]);
    free_ddmatrix(tmp_mat[0]);
    free_ddmatrix(tmp_mat[1]);
}



// absmax_col_ddmatrix
void absmax_col_ddmatrix(double mu[DDSIZE], long int *max_i, long int col_index, DDMatrix mat)
{
    long int i, max_index = 0;
    double abs_aij[DDSIZE];

	//mu = fabs(mat[0 * col_dim + j]);
    rdd_abs(mu, get_ddmatrix_ij(mat, 0, col_index));
	for(i = 1; i < mat->real_row_dim; i++)
	{
		rdd_abs(abs_aij, get_ddmatrix_ij(mat, i, col_index));
		if(rdd_cmp(abs_aij, mu) > 0)
        {
			rdd_set(mu, abs_aij);
            max_index = i;
        }
	}

    if(max_i != NULL)
        *max_i = max_index;

    return;
}

/*------------------------------------------------------------------------------*/
/* SplitMat_B: same as split_ddmatrix_dmat, but the threshold of the second      */
/* operand is taken over columns, so the maxima are reduced per column.  The     */
/* sweep still walks the matrix row-wise -- one partial maximum vector per       */
/* thread, combined afterwards -- because the column-wise walk the definition    */
/* suggests would touch a new cache line on every element.                       */
/*------------------------------------------------------------------------------*/
// return real_num_div
/*------------------------------------------------------------------------------*/
/* SplitMat_B: same as split_ddmatrix_dmat_ex(), but the threshold and the    */
/* scale of the second operand are taken over columns, so the maxima are reduced */
/* per column -- one partial maximum vector per thread, combined afterwards, so  */
/* that the sweep can still walk the matrix row-wise.                            */
/*------------------------------------------------------------------------------*/
int split_ddmatrix_t_dmat_ex(DMatrix ret_mat[], long int col_shift[], int num_div, DDMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, org_stride, ret_stride;
	int real_num_div, num_digits = 53, num_threads, thread; // IEEE double prec.
	double mu_total, tail_exp;
	DDMatrix tmp_org_mat;
	DMatrix own_ret_mat = NULL, in_ret_mat;
	double *power2, *mu_local; // 2^t_exp of each column, per-thread column maxima
	long int *shift;

	row_dim = org_mat->row_dim;
	col_dim = org_mat->col_dim;

	if(ret_mat != NULL)
	{
		if((ret_mat[0]->row_dim != row_dim) || (ret_mat[0]->col_dim != col_dim))
		{
			fprintf(stderr, "ERROR: split_ddmatrix_t_dmat\n");
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
		fprintf(stderr, "ERROR: split_ddmatrix_t_dmat: cannot allocate\n");
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

	tmp_org_mat = init_ddmatrix(row_dim, col_dim);
	subst_ddmatrix(tmp_org_mat, org_mat);
	org_stride = tmp_org_mat->real_col_dim;

	// the column-independent half of t_exp
	tail_exp = ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);

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
			double *org_row[DDSIZE];
			double s, high_ij, scaled_ij;
			double org_ij[DDSIZE], high_dd[DDSIZE], rest_ij[DDSIZE];
			long int sigma;
			int comp;

			for(comp = 0; comp < DDSIZE; comp++)
			{
				org_row[comp] = tmp_org_mat->element[comp] + i * org_stride;
				high_dd[comp] = 0.0;
			}

			for(j = 0; j < col_dim; j++)
			{
				sigma = (shift != NULL) ? shift[j] : 0;
				s = power2[j];

				scaled_ij = (sigma != 0) ? bnc_oz_ldexp(org_row[0][j], -sigma) : org_row[0][j];

				high_ij = scaled_ij + s;
				high_ij = high_ij - s;
				ret_row[j] = high_ij;

				for(comp = 0; comp < DDSIZE; comp++)
					org_ij[comp] = org_row[comp][j];
				high_dd[0] = (sigma != 0) ? bnc_oz_ldexp(high_ij, sigma) : high_ij;

				rdd_sub(rest_ij, org_ij, high_dd);

				for(comp = 0; comp < DDSIZE; comp++)
					org_row[comp][j] = rest_ij[comp];
			}
		}

		real_num_div = index + 1;
	}

	free(power2);
	free(mu_local);
	free_ddmatrix(tmp_org_mat);
	if(own_ret_mat != NULL)
		free_dmatrix(own_ret_mat);

	return real_num_div;
}

// SplitMat_B without the scaling; kept for callers that cannot apply a scale factor
int split_ddmatrix_t_dmat(DMatrix ret_mat[], int num_div, DDMatrix org_mat)
{
	return split_ddmatrix_t_dmat_ex(ret_mat, NULL, num_div, org_mat);
}

// SplitMat_B
void split_ddmatrix_t2(DMatrix ret_mat[], int num_div, DDMatrix org_mat, int num_digits)
{
	long int i, j, index, row_dim, col_dim;
	//int flag_stop = 0, num_digits = 53; // IEEE double prec.
	int flag_stop = 0; //, num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//int flag_stop = 0, num_digits = 64; // IEEE double prec.
	//double *s;
    DDMatrix tmp_org_mat;
    DDMatrix s, tmp_mat[2];
	double mu[DDSIZE], abs_aij[DDSIZE], t_exp[DDSIZE], power2[DDSIZE], two[DDSIZE] = {2.0, 0.0};
	//double mu, abs_aij, t_exp, power2;

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;

	// initialize s
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    s = init_ddmatrix(row_dim, col_dim);
    //s = init_dmatrix(row_dim, col_dim);

    tmp_mat[0] = init_ddmatrix(row_dim, col_dim);
    tmp_mat[1] = init_ddmatrix(row_dim, col_dim);

    tmp_org_mat = init_ddmatrix(row_dim, col_dim);
    subst_ddmatrix(tmp_org_mat, org_mat);

    for(index = 0; index < num_div; index++)
    {
        //subst_dmatrix_ddmat(ret_mat[index], tmp_org_mat);
        // mu[j] = max_j |mat[i, j]|
        for(j = 0; j < col_dim; j++)
        {
            /* mu = fabs(mat[0 * col_dim + j]);
            for(i = 1; i < row_dim; i++)
            {
                abs_aij = fabs(mat[i * col_dim + j]);
                if(abs_aij > mu)
                    mu = abs_aij;
            }
            */
            absmax_col_ddmatrix(mu, NULL, j, tmp_org_mat);
            //absmax_col_ddmatrix(mu, NULL, j, ret_mat[index]);
            //printf("mu%d: %15.7e ", j, mu);

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);
            t_exp[1] = 0.0;
            //t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);
            if(isnan(t_exp[0]))
                flag_stop = 1;

            // s[i, j] = 2^t_exp
            rdd_pow(power2, two, t_exp);
            rdd_pow_mpfr(power2, two, t_exp);
            //power2 = pow(2.0, t_exp);

            //printf("index, j, power2 = %ld, %ld, %25.17e\n", index, j, power2[0]);
            for(i = 0; i < row_dim; i++)
                set_ddmatrix_ij(s, i, j, power2);
                //s[i * col_dim + j] = pow(2.0, t_exp);
        }
        if(flag_stop == 1)
            break;

        // tmp_mat := mat + s
        //add_dmatrix(tmp_mat[0], ret_mat[index], s);
        add_ddmatrix(tmp_mat[0], tmp_org_mat, s);

        // high_mat := tmp_mat - s
        sub_ddmatrix(tmp_mat[1], tmp_mat[0], s);
        subst_dmatrix_ddmat(ret_mat[index], tmp_mat[1]);

        // low_mat := mat - high_mat
        //sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);
        sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
    }

    // free s
	free_ddmatrix(s);
    free_ddmatrix(tmp_mat[0]);
    free_ddmatrix(tmp_mat[1]);
    free_ddmatrix(tmp_org_mat);
}

/*------------------------------------------------------------------------------*/
/* Matrix multiplication based on Ozaki scheme                                   */
/*                                                                               */
/* ret = sum_{i,j} div_a[i] * div_b[j].  In BNC_OZ_GEMM_MODE_OWN the rows of ret */
/* are cut into blocks and one thread takes a block at a time, running every     */
/* slice product for it with a single-threaded DGEMM and accumulating in DD on   */
/* the spot; the DD accumulation, which a threaded BLAS would leave serial, is   */
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
/* DGEMM and accumulating in DD on the spot; the accumulation, which a        */
/* threaded BLAS would leave serial, is thereby parallelized as well and the     */
/* block of ret stays hot in cache across all the slice pairs.  Blocks are       */
/* disjoint and each element of ret still sums its slice products in the         */
/* original order, so the result does not depend on the number of threads.       */
/* BNC_OZ_GEMM_MODE_BLAS keeps the old shape (one full DGEMM per slice pair,     */
/* left to the BLAS to thread).                                                  */
/*------------------------------------------------------------------------------*/
void mul_ddmatrix_oz(DDMatrix ret, DDMatrix a, int max_num_div_a, DDMatrix b, int max_num_div_b)
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
        fprintf(stderr, "ERROR: mul_ddmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
        return;
    }

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_b = (DMatrix *)calloc(max_num_div_b, sizeof(DMatrix));
    row_shift = (long int *)calloc((size_t)max_num_div_a * (size_t)row_dim, sizeof(long int));
    col_shift = (long int *)calloc((size_t)max_num_div_b * (size_t)col_dim, sizeof(long int));
    if(div_a == NULL || div_b == NULL || row_shift == NULL || col_shift == NULL)
    {
        fprintf(stderr, "ERROR: mul_ddmatrix_oz: cannot allocate\n");
        free(div_a); free(div_b); free(row_shift); free(col_shift);
        return;
    }
    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, mid_dim);
    for(i = 0; i < max_num_div_b; i++)
        div_b[i] = init_dmatrix(mid_dim, col_dim);

    real_num_div_a = split_ddmatrix_dmat_ex(div_a, row_shift, max_num_div_a, a);
    real_num_div_b = split_ddmatrix_t_dmat_ex(div_b, col_shift, max_num_div_b, b);

    set0_ddmatrix(ret);

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
            double *ret_row[DDSIZE];
            double ret_ij[DDSIZE], add_ij[DDSIZE], sum_ij[DDSIZE];

            for(comp = 0; comp < DDSIZE; comp++)
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

                            for(comp = 0; comp < DDSIZE; comp++)
                                ret_row[comp] = ret->element[comp] + (row_start + ii) * ret->real_col_dim;

                            for(jj = 0; jj < col_dim; jj++)
                            {
                                for(comp = 0; comp < DDSIZE; comp++)
                                    ret_ij[comp] = ret_row[comp][jj];
                                add_ij[0] = bnc_oz_ldexp(buf_row[jj], shift_a + shift_b[jj]);

                                rdd_add(sum_ij, ret_ij, add_ij);

                                for(comp = 0; comp < DDSIZE; comp++)
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
        double *ret_row[DDSIZE];
        double ret_ij[DDSIZE], add_ij[DDSIZE], sum_ij[DDSIZE];

        for(comp = 0; comp < DDSIZE; comp++)
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

                    for(comp = 0; comp < DDSIZE; comp++)
                        ret_row[comp] = ret->element[comp] + ii * ret->real_col_dim;

                    for(jj = 0; jj < col_dim; jj++)
                    {
                        for(comp = 0; comp < DDSIZE; comp++)
                            ret_ij[comp] = ret_row[comp][jj];
                        add_ij[0] = bnc_oz_ldexp(buf_row[jj], shift_a + shift_b[jj]);

                        rdd_add(sum_ij, ret_ij, add_ij);

                        for(comp = 0; comp < DDSIZE; comp++)
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

/*------------------------------------------------------------------------------*/
/* Matrix-Vector multiplication based on Ozaki scheme                            */
/*                                                                               */
/* Same blocking as mul_ddmatrix_oz: a thread owns a range of rows of ret and    */
/* runs all slice pairs for it, so the DD accumulation is parallel too.          */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
/* Matrix-Vector multiplication based on Ozaki scheme                            */
/*                                                                               */
/* Same blocking and the same exponent handling as mul_ddmatrix_oz(): a       */
/* thread owns a range of rows of ret and runs all slice pairs for it.           */
/*------------------------------------------------------------------------------*/
void mul_ddmatrix_ddvec_oz(DDVector ret, DDMatrix a, int max_num_div_a, DDVector vb, int max_num_div_vb)
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
        fprintf(stderr, "ERROR: mul_ddmatrix_ddvec_oz: cannot allocate\n");
        free(div_a); free(div_vb); free(row_shift); free(vec_shift);
        return;
    }

    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, col_dim);
    for(i = 0; i < max_num_div_vb; i++)
        div_vb[i] = init_dvector(vb->dim); // vb->dim, not ret->dim: they differ when a is not square

    real_num_div_a = split_ddmatrix_dmat_ex(div_a, row_shift, max_num_div_a, a);
    real_num_div_vb = split_ddvector_dvec_ex(div_vb, vec_shift, max_num_div_vb, vb);

    set0_ddvector(ret);

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
            double ret_i[DDSIZE], add_i[DDSIZE], sum_i[DDSIZE];

            for(comp = 0; comp < DDSIZE; comp++)
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

                            for(comp = 0; comp < DDSIZE; comp++)
                                ret_i[comp] = ret->element[comp][row_start + ii];
                            add_i[0] = bnc_oz_ldexp(buf[ii], shift_a + vec_shift[div_j]);

                            rdd_add(sum_i, ret_i, add_i);

                            for(comp = 0; comp < DDSIZE; comp++)
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
        double ret_i[DDSIZE], add_i[DDSIZE], sum_i[DDSIZE];

        for(comp = 0; comp < DDSIZE; comp++)
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

                    for(comp = 0; comp < DDSIZE; comp++)
                        ret_i[comp] = ret->element[comp][ii];
                    add_i[0] = bnc_oz_ldexp(div_ret->element[ii], shift_a + vec_shift[j]);

                    rdd_add(sum_i, ret_i, add_i);

                    for(comp = 0; comp < DDSIZE; comp++)
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
void mul_cddmatrix_oz_3m(CDDMatrix ret, CDDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    DDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_ddmatrix(b->re->row_dim, ret->re->col_dim);

    mul_ddmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    mul_ddmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_b_image);
    sub_ddmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_ddmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_ddmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        add_ddmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        add_ddmatrix(t3, a->re, a->im);
        add_ddmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_ddmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_ddmatrix(ret->im, ret->im, t1);
        sub_ddmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void mul_cddmatrix_oz_4m(CDDMatrix ret, CDDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    DDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_ddmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    mul_ddmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_b_image);
    sub_ddmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_ddmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_ddmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_b_image);
        add_ddmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_ddmatrix(t3, a->re, a->im);
        add_ddmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_ddmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_ddmatrix(ret->im, ret->im, t1);
        sub_ddmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// Testing
#ifdef DEBUG

//#define DIM 5
//#define DIM 10
//#define DIM 100
#define DIM 512
//#define DIM 1024


//#define MAX_NUM_DIV 2
//#define MAX_NUM_DIV 5
//#define MAX_NUM_DIV 6
//#define MAX_NUM_DIV 10
#define MAX_NUM_DIV 128

//int main()
int main(int argc, char *argv[])
{
    long int i, j, dim, num_div;
    int num_digits;
    int real_num_div;
    double tmp[DDSIZE], tmp2[DDSIZE];
    DDVector vec_org, vec, vec_b, vec_c;
    DVector dvec[MAX_NUM_DIV];
    DDMatrix mat_a, mat_b, mat_c, mat;
    DMatrix dmat[MAX_NUM_DIV];
    ddfloat ddtmp, ddtmp2;

    if(argc <= 1)
    {
        printf("$ %s [dim] [num_div] [num_digits]\n", argv[0]);
        return 0;
    }
    dim = atoi(argv[1]);
    num_div = MAX_NUM_DIV;
    if(argc >= 3)
        num_div = atoi(argv[2]);
    num_digits = 53; //SPLIT_NUM_DIGITS;
    if(argc >= 4)
        num_digits = atoi(argv[3]);
    //goto ddmatrix;

// DDVector 
    vec_org = init_ddvector(dim);
    vec = init_ddvector(dim);
    for(i = 0; i < num_div; i++)
        dvec[i] = init_dvector(dim);


    // set random
    srand(dim);
    for(i = 0; i < dim; i++)
    {
        rdd_set_d(tmp, (double)rand());
        rdd_mul_d(tmp, tmp, (double)rand());
        set_ddvector_i(vec_org, i, tmp);
    }

    // split
    real_num_div = split_ddvector_dvec(dvec, num_div, vec_org); //, 53);

    //for(i = 0; i < num_div; i++)
    for(i = 0; i < real_num_div; i++)
    {
        //printf("dvec[%d]:\n", i); print_dvector(dvec[i]);

        add_ddvector_dvec(vec, vec, dvec[i]);
    }
    //sub_ddvector(vec, vec, vec_org);
    //printf("vec - vec_org:\n"); print_ddvector(vec);
    //printf("\n");
    for(i = 0; i < dim; i++)
    {
        ddtmp = get_ddvector_i_ddfloat(vec, i);
        ddtmp2 = get_ddvector_i_ddfloat(vec_org, i);
        printf("%5d %25.17e %25.17e\n%5d %25.17e %25.17e\n", i, ddtmp.val[0], ddtmp.val[1], i, ddtmp2.val[0], ddtmp2.val[1]);
    }
    

    free_ddvector(vec_org);
    free_ddvector(vec);
    for(i = 0; i < num_div; i++)
        free_dvector(dvec[i]);

    //sgoto end;
// DMatrix
ddmatrix:
    vec_b = init_ddvector(dim);
    vec_c = init_ddvector(dim);
    vec = init_ddvector(dim);

    srand(dim);
    for(i = 0; i < dim; i++)
    {
        rdd_set_d(tmp, (double)rand());
        rdd_mul_d(tmp, tmp, (double)rand());
        set_ddvector_i(vec_b, i, tmp);
    }

    for(i = 0; i < num_div; i++)
        dvec[i] = init_dvector(dim);

    mat_a = init_ddmatrix(dim, dim);
    mat_b = init_ddmatrix(dim, dim);
    mat_c = init_ddmatrix(dim, dim);
    mat = init_ddmatrix(dim, dim);
    for(i = 0; i < num_div; i++)
        dmat[i] = init_dmatrix(dim, dim);

    // set random
    srand(dim);
    for(i = 0; i < dim; i++)
    {
        for(j = 0; j < dim; j++)
        {
            //rdd_set_d(tmp, (double)rand() / (double)RAND_MAX);
            rdd_set_d(tmp, (double)rand());
            rdd_sqrt(tmp, tmp);
            rdd_mul_d(tmp, tmp, (double)rand());
            set_ddmatrix_ij(mat_a, i, j, tmp);

            rdd_set_d(tmp, (double)rand() / (double)RAND_MAX);
            rdd_mul_d(tmp, tmp, (double)rand());
            set_ddmatrix_ij(mat_b, i, j, tmp);
        }
    }

    // split_A
    printf("Split_A:");
    real_num_div = split_ddmatrix_dmat(dmat, num_div, mat_a); //, num_digits);
    printf("%d -> %d mats\n", num_div, real_num_div);

    for(i = 0; i < real_num_div; i++)
    {
        //printf("dmat[%d]:\n", i); print_dmatrix(dmat[i]);

        add_ddmatrix_dmat(mat, mat, dmat[i]);
    }
    sub_ddmatrix(mat, mat, mat_a);
    printf("||mat - mat_org|| / ||mat_org||:\n"); 
    normf_ddmatrix(tmp, mat);
    normf_ddmatrix(tmp2, mat_a);
    rdd_div(tmp, tmp, tmp2);
    rdd_out_str(tmp); //print_ddmatrix(mat);
    printf("\n"); 

    // split_B
    printf("Split_B:");
    real_num_div = split_ddmatrix_t_dmat(dmat, num_div, mat_b); //, num_digits);
    printf("%d -> %d mats\n", num_div, real_num_div);

    for(i = 0; i < real_num_div; i++)
    {
        //printf("dmat[%d]:\n", i); print_dmatrix(dmat[i]);

        add_ddmatrix_dmat(mat, mat, dmat[i]);
    }
    sub_ddmatrix(mat, mat, mat_b);
    printf("||mat - mat_org|| / ||mat_org||:\n"); 
    normf_ddmatrix(tmp, mat);
    normf_ddmatrix(tmp2, mat_a);
    rdd_div(tmp, tmp, tmp2);
    rdd_out_str(tmp); //print_ddmatrix(mat);
    printf("\n"); 

    // a * b
    mul_ddmatrix_oz(mat, mat_a, num_div, mat_b, num_div); // , num_digits);
    mul_ddmatrix(mat_c, mat_a, mat_b);

    //printf("split_bits, dim, num_div = %d, %ld, %ld\n", num_digits, dim, num_div);
    printf("split_bits, dim, num_div_a, num_div_b = %d, %ld, %ld\n", 53, dim, num_div);
    sub_ddmatrix(mat, mat, mat_c);
    printf("||mat_oz - mat_org|| / ||mat_org||:\n"); 
    normf_ddmatrix(tmp, mat);
    normf_ddmatrix(tmp2, mat_c);
    rdd_div(tmp, tmp, tmp2);
    rdd_out_str(tmp); //print_ddmatrix(mat); 
    printf("\n"); 

    // a * vb
    mul_ddmatrix_ddvec_oz(vec_c, mat_a, num_div, vec_b, num_div);
    mul_ddmatrix_ddvec(vec, mat_a, vec_b);

    //printf("split_bits, dim, num_div = %d, %ld, %ld\n", num_digits, dim, num_div);
    //printf("split_bits, dim, num_div_a, num_div_b = %d, %ld, %ld\n", 53, dim, num_div);
    sub_ddvector(vec, vec, vec_c);
    printf("||vec_oz - vec_org|| / ||vec_org||:\n"); 
    norm2_ddvector(tmp, vec);
    norm2_ddvector(tmp2, vec_c);
    rdd_div(tmp, tmp, tmp2);
    rdd_out_str(tmp); //print_ddmatrix(mat); 
    printf("\n"); 

    free_ddmatrix(mat_a);
    free_ddmatrix(mat_b);
    free_ddmatrix(mat_c);
    free_ddmatrix(mat);
    for(i = 0; i < num_div; i++)
        free_dmatrix(dmat[i]);

    free_ddvector(vec_b);
    free_ddvector(vec_c);
    free_ddvector(vec);
    for(i = 0; i < num_div; i++)
        free_dvector(dvec[i]);

end:
    return 0;
}

// Testing
#endif // DEBUG
