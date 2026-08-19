/********************************************************************************/
/* mpf_oz_scheme: Multiple precision linear computation based on Ozaki scheme.  */
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
//#include "rdd.h" // [dtq]dfloat and its arithmetic r[dtq]d_*
#include "matmul_strassen.h"

#include <stdlib.h>
#include <limits.h> // LONG_MIN, for the "no exponent" marker of an all-zero row

/*------------------------------------------------------------------------------*/
/* Exponent handling of the mpf_t split.                                         */
/*                                                                               */
/* mpf_t reaches 2^+-2^30 while the slices are plain doubles, so nothing can be  */
/* converted before it has been scaled: mpf_get_d() alone returns +-Inf above    */
/* DBL_MAX and 0 below DBL_MIN, and the threshold 2^(ceil(log2 mu) + s) leaves   */
/* the double range long before that.  These two helpers do the scaled           */
/* conversion in both directions; both are exact, because a power of two is.     */
/*------------------------------------------------------------------------------*/

// exponent e of x with |x| in [2^(e-1), 2^e); LONG_MIN for zero
long int bnc_oz_mpf_exp2(mpf_srcptr x)
{
    long int exponent = 0;

    if(mpf_sgn(x) == 0)
        return LONG_MIN;

    mpf_get_d_2exp(&exponent, x); // x = mantissa * 2^exponent, |mantissa| in [1/2, 1)

    return exponent;
}

// 2^-shift * x as a double; flushes to zero when that is far below the shift
double bnc_oz_mpf_get_scaled_d(mpf_srcptr x, long int shift)
{
    long int exponent = 0;
    double mantissa;

    if(mpf_sgn(x) == 0)
        return 0.0;

    mantissa = mpf_get_d_2exp(&exponent, x);

    return bnc_oz_ldexp(mantissa, exponent - shift);
}

// ret := 2^shift * value, exactly
void bnc_oz_mpf_set_scaled_d(mpf_ptr ret, double value, long int shift)
{
    mpf_set_d(ret, value);

    if(shift > 0)
        mpf_mul_2exp(ret, ret, (unsigned long)shift);
    else if(shift < 0)
        mpf_div_2exp(ret, ret, (unsigned long)(-shift));
}


//#ifndef DEBUG // not DEBUG
// absmax_mpfvector
void absmax_mpfvector(mpf_t ret, long int *max_index, MPFVector vec)
{
    long int i, max_i, dim = vec->dim;
    mpf_t abs_val;

    // initialize
    mpf_init2(abs_val, mpf_get_prec(ret));

    max_i = 0;
    //ret[0] = 0.0; ret[1] = 0.0;
    mpf_set_ui(ret, 0UL);

    for(i = 0; i < dim; i++)
    {
        //abs_val = fabs(get_mpfvector_i(vec, i));
        mpf_abs(abs_val, get_mpfvector_i(vec, i));
        if(mpf_cmp(ret, abs_val) < 0)
        {
            mpf_set(ret, abs_val);
            max_i = i;
        }
    }

    if(max_index != NULL)
        *max_index = max_i;

    // free
    mpf_clear(abs_val);

    return;
}

/* c = a + (double)b */
void add_mpfvector_dvec(MPFVector c, MPFVector a, DVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_mpfvector_dvec\n");
		return;
	}

	mpf_t tmp;

    mpf_init2(tmp, c->prec);

	for(i = 0; i < c->dim; i++)
	{
		mpf_add_d(tmp, get_mpfvector_i(a, i), get_dvector_i(b, i));
		set_mpfvector_i(c, i, tmp);
	}

    mpf_clear(tmp);
}

/* c = a - (double)b */
void sub_mpfvector_dvec(MPFVector c, MPFVector a, DVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_mpfvector_dvec\n");
		return;
	}

	mpf_t tmp;

    mpf_init2(tmp, c->prec);

	for(i = 0; i < c->dim; i++)
	{
		mpf_sub_d(tmp, get_mpfvector_i(a, i),  get_dvector_i(b, i));
		set_mpfvector_i(c, i, tmp);
	}

    mpf_clear(tmp);
}

// extract vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
//void extract_mpfvector(DVector ret_vec[], int num_div, mpfvector org_vec, int num_bits)
/*------------------------------------------------------------------------------*/
/* split vector: 2^shift[index] * (ret_vec[0] + ... ) = org_vec                  */
/*                                                                               */
/* The vector plays the role of a single column of B, so it carries one exponent */
/* per slice.  shift may be NULL; see oz_scheme.h.                               */
/*------------------------------------------------------------------------------*/
int split_mpfvector_dvec_ex(DVector ret_vec[], long int shift[], int num_div, MPFVector org_vec)
{
    unsigned int prec = org_vec->prec;
    long int dim = org_vec->dim, i;
    int index, real_num_div, num_bits = 53; // IEEE754 binary64
    double org_vec_i, high_i, absmax_org_vec, threshold, tail_exp;
    long int sigma, vec_exp, element_exp;
    mpf_t tmp;
    MPFVector tmp_org_vec;
    DVector own_ret_vec = NULL, in_ret_vec;

    mpf_init2(tmp, prec);

    if(ret_vec == NULL)
    {
        own_ret_vec = init_dvector(dim);
        in_ret_vec = own_ret_vec;
    }
    else
        in_ret_vec = ret_vec[0];

    // tmp_org_vec := org_vec
    tmp_org_vec = init2_mpfvector(dim, prec);
    subst_mpfvector(tmp_org_vec, org_vec);

    tail_exp = ceil(((double)num_bits + DLOG2((double)(dim))) / 2.0);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        if(ret_vec != NULL)
            in_ret_vec = ret_vec[index];

        sigma = 0;
        if(shift != NULL)
        {
            vec_exp = LONG_MIN;
            for(i = 0; i < dim; i++)
            {
                element_exp = bnc_oz_mpf_exp2(get_mpfvector_i(tmp_org_vec, i));
                if(element_exp > vec_exp)
                    vec_exp = element_exp;
            }
            if(vec_exp != LONG_MIN)
                sigma = vec_exp;
            shift[index] = sigma;
        }

        absmax_org_vec = 0.0;
        for(i = 0; i < dim; i++)
        {
            org_vec_i = (sigma != 0) ? bnc_oz_mpf_get_scaled_d(get_mpfvector_i(tmp_org_vec, i), sigma)
                                     : mpf_get_d(get_mpfvector_i(tmp_org_vec, i));
            set_dvector_i(in_ret_vec, i, org_vec_i);

            if(fabs(org_vec_i) > absmax_org_vec)
                absmax_org_vec = fabs(org_vec_i);
        }

        // vector is exhausted
        if(absmax_org_vec == 0.0) break;

        threshold = pow(2.0, ceil(DLOG2(absmax_org_vec)) + tail_exp);

        for(i = 0; i < dim; i++)
        {
            org_vec_i = get_dvector_i(in_ret_vec, i);
            high_i = org_vec_i + threshold;
            high_i = high_i - threshold;
            set_dvector_i(in_ret_vec, i, high_i);

            // low vector := vector - 2^shift * high
            bnc_oz_mpf_set_scaled_d(tmp, high_i, sigma);
            mpf_sub(tmp, get_mpfvector_i(tmp_org_vec, i), tmp);
            set_mpfvector_i(tmp_org_vec, i, tmp);
        }

        real_num_div = index + 1;
    }

    mpf_clear(tmp);
    free_mpfvector(tmp_org_vec);
    if(own_ret_vec != NULL)
        free_dvector(own_ret_vec);

    return real_num_div;
}

// split vector without the scaling
int split_mpfvector_dvec(DVector ret_vec[], int num_div, MPFVector org_vec)
{
    return split_mpfvector_dvec_ex(ret_vec, NULL, num_div, org_vec);
}

// absmax_row_mpfmatrix
void absmax_row_mpfmatrix(mpf_t mu, long int *max_j, long int row_index, MPFMatrix mat)
{
    long int j, max_index = 0;
    mpf_t abs_aij;

    mpf_init2(abs_aij, mpf_get_prec(mu));

	//mu = fabs(mat[i * col_dim + 0]);
    mpf_abs(mu, get_mpfmatrix_ij(mat, row_index, 0));

	for(j = 1; j < mat->col_dim; j++)
	{
		mpf_abs(abs_aij, get_mpfmatrix_ij(mat, row_index, j));
		if(mpf_cmp(abs_aij, mu) > 0)
        {
			//mu = abs_aij;
            mpf_set(mu, abs_aij);
            max_index = j;
        }
	}

    if(max_j != NULL)
        *max_j = max_index;

    mpf_clear(abs_aij);

    //return mu;
    return;
}

/* c := a + (doble)b */
void add_mpfmatrix_dmat(MPFMatrix c, MPFMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim;
	int num_threads;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_mpfmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_mpfmatrix_dmat\n");
		return;
	}
	col_dim = c->col_dim;

	num_threads = bnc_oz_get_num_threads();

	// row-wise: the rows are independent, and every thread carries its own
	// mpf_t scratch because one shared variable cannot be written concurrently
#ifdef _OPENMP
	#pragma omp parallel num_threads(num_threads) private(i, j)
#endif // _OPENMP
	{
		mpf_t tmp;

		mpf_init2(tmp, c->prec);

#ifdef _OPENMP
		#pragma omp for schedule(static)
#endif // _OPENMP
		for(i = 0; i < row_dim; i++)
		{
			for(j = 0; j < col_dim; j++)
			{
				mpf_add_d(tmp, get_mpfmatrix_ij(a, i, j), get_dmatrix_ij(b, i, j));
				set_mpfmatrix_ij(c, i, j, tmp);
			}
		}

		mpf_clear(tmp);
	}
}

/* c := a - (doble)b */
void sub_mpfmatrix_dmat(MPFMatrix c, MPFMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim;
	int num_threads;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_mpfmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_mpfmatrix_dmat\n");
		return;
	}
	col_dim = c->col_dim;

	num_threads = bnc_oz_get_num_threads();

	// row-wise: the rows are independent, and every thread carries its own
	// mpf_t scratch because one shared variable cannot be written concurrently
#ifdef _OPENMP
	#pragma omp parallel num_threads(num_threads) private(i, j)
#endif // _OPENMP
	{
		mpf_t tmp;

		mpf_init2(tmp, c->prec);

#ifdef _OPENMP
		#pragma omp for schedule(static)
#endif // _OPENMP
		for(i = 0; i < row_dim; i++)
		{
			for(j = 0; j < col_dim; j++)
			{
				mpf_sub_d(tmp, get_mpfmatrix_ij(a, i, j), get_dmatrix_ij(b, i, j));
				set_mpfmatrix_ij(c, i, j, tmp);
			}
		}

		mpf_clear(tmp);
	}
}

//#define SPLIT_NUM_DIGITS 64

// SplitMat_A
/*------------------------------------------------------------------------------*/
/* SplitMat_A: ret_mat[0] + ret_mat[1] + ... = org_mat, every ret_mat[] a plain  */
/* double matrix whose rows multiply without rounding.                           */
/*                                                                               */
/* Two passes over the data per split instead of the six the straightforward     */
/* formulation needs, and both run row-parallel because rows never interact.     */
/* The threshold is constant along a row, so a vector of row thresholds replaces */
/* the full s matrix that used to be built and streamed for every split.         */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
/* SplitMat_A: 2^row_shift[index][i] * (ret_mat[0] + ret_mat[1] + ...) = org_mat */
/*                                                                               */
/* Two row-parallel passes per split, with the row normalized by a power of two  */
/* first so that the conversion to double and the threshold both stay inside the */
/* double exponent range whatever the mpf_t exponents are.  A fresh exponent is  */
/* taken for every slice, so the dynamic range a row may span is bounded by      */
/* num_div alone.  row_shift may be NULL, which asks for the old unscaled split  */
/* and its restriction to operands near 2^0; see oz_scheme.h.                    */
/*------------------------------------------------------------------------------*/
int split_mpfmatrix_dmat_ex(DMatrix ret_mat[], long int row_shift[], int num_div, MPFMatrix org_mat)
{
	unsigned long prec = org_mat->prec;
	long int i, j, index, row_dim, col_dim, ret_stride;
	long int num_digits = 53; // IEEE double prec.
	int real_num_div, num_threads;
	double mu_total, tail_exp;
	MPFMatrix tmp_org_mat;
	DMatrix own_ret_mat = NULL, in_ret_mat;
	double *power2, *mantissa_buf; // 2^t_exp of each row, cached mantissas
	long int *shift, *exponent_buf;

	row_dim = org_mat->row_dim;
	col_dim = org_mat->col_dim;

	if(ret_mat != NULL)
	{
		if((ret_mat[0]->row_dim != row_dim) || (ret_mat[0]->col_dim != col_dim))
		{
			fprintf(stderr, "ERROR: split_mpfmatrix_dmat\n");
			return 0;
		}
	}

	num_threads = bnc_oz_get_num_threads();

	// one row of mantissas and exponents per thread: the row maximum has to be
	// known before anything can be converted, and caching what mpf_get_d_2exp()
	// already returned halves the number of conversions
	power2 = (double *)calloc((size_t)row_dim, sizeof(double));
	mantissa_buf = (double *)calloc((size_t)num_threads * (size_t)col_dim, sizeof(double));
	exponent_buf = (long int *)calloc((size_t)num_threads * (size_t)col_dim, sizeof(long int));
	if(mantissa_buf == NULL || exponent_buf == NULL) // retry single-threaded rather than give up
	{
		num_threads = 1;
		free(mantissa_buf);
		free(exponent_buf);
		mantissa_buf = (double *)calloc((size_t)col_dim, sizeof(double));
		exponent_buf = (long int *)calloc((size_t)col_dim, sizeof(long int));
	}
	if(power2 == NULL || mantissa_buf == NULL || exponent_buf == NULL)
	{
		fprintf(stderr, "ERROR: split_mpfmatrix_dmat: cannot allocate\n");
		free(power2);
		free(mantissa_buf);
		free(exponent_buf);
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
	tmp_org_mat = init2_mpfmatrix(row_dim, col_dim, prec);
	subst_mpfmatrix(tmp_org_mat, org_mat);

	// the row-independent half of t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim) / 2)
	tail_exp = ceil(((double)num_digits + DLOG2((double)col_dim)) / 2.0);

	real_num_div = 0;
	for(index = 0; index < num_div; index++)
	{
		if(ret_mat != NULL)
			in_ret_mat = ret_mat[index];
		ret_stride = in_ret_mat->real_col_dim;
		shift = (row_shift != NULL) ? (row_shift + (size_t)index * (size_t)row_dim) : NULL;

		// pass 1: ret_mat[index] := 2^-shift[i] * tmp_org_mat, mu[i] = max_j |...|
		mu_total = 0.0;

#ifdef _OPENMP
		#pragma omp parallel for num_threads(num_threads) schedule(static) private(i, j) reduction(+:mu_total)
#endif // _OPENMP
		for(i = 0; i < row_dim; i++)
		{
			double *ret_row = in_ret_mat->element + i * ret_stride;
			double mu = 0.0, org_ij, abs_org_ij;
			long int sigma = 0, row_exp = LONG_MIN;
			double *mantissa;
			long int *exponent;
			mpf_ptr org_ptr;

#ifdef _OPENMP
			mantissa = mantissa_buf + (size_t)omp_get_thread_num() * (size_t)col_dim;
			exponent = exponent_buf + (size_t)omp_get_thread_num() * (size_t)col_dim;
#else // _OPENMP
			mantissa = mantissa_buf;
			exponent = exponent_buf;
#endif // _OPENMP

			// the exponent of the largest entry, taken in mpf_t so that entries
			// outside the double range still count
			for(j = 0; j < col_dim; j++)
			{
				org_ptr = get_mpfmatrix_ij(tmp_org_mat, i, j);

				if(mpf_sgn(org_ptr) == 0)
				{
					mantissa[j] = 0.0;
					exponent[j] = 0;
					continue;
				}

				mantissa[j] = mpf_get_d_2exp(&(exponent[j]), org_ptr);
				if(exponent[j] > row_exp)
					row_exp = exponent[j];
			}

			if(shift != NULL)
			{
				if(row_exp != LONG_MIN)
					sigma = row_exp;
				shift[i] = sigma;
			}

			for(j = 0; j < col_dim; j++)
			{
				org_ij = (mantissa[j] != 0.0) ? bnc_oz_ldexp(mantissa[j], exponent[j] - sigma) : 0.0;
				ret_row[j] = org_ij;

				abs_org_ij = fabs(org_ij);
				if(abs_org_ij > mu)
					mu = abs_org_ij;
			}

			// s[i, j] = 2^t_exp
			power2[i] = pow(2.0, ceil(DLOG2(mu)) + tail_exp);
			mu_total += mu;
		}

		// nothing left to split
		if(mu_total == 0.0) break;

		// pass 2: high := (mat + s) - s, and mat := mat - 2^shift[i] * high
#ifdef _OPENMP
		#pragma omp parallel num_threads(num_threads) private(i, j)
#endif // _OPENMP
		{
			mpf_t tmp;

			mpf_init2(tmp, prec);

#ifdef _OPENMP
			#pragma omp for schedule(static)
#endif // _OPENMP
			for(i = 0; i < row_dim; i++)
			{
				double *ret_row = in_ret_mat->element + i * ret_stride;
				double s = power2[i], high_ij;
				long int sigma = (shift != NULL) ? shift[i] : 0;

				for(j = 0; j < col_dim; j++)
				{
					// (x + s) - s keeps the leading bits of x; valid under the IEEE
					// semantics this library is compiled with (no -ffast-math)
					high_ij = ret_row[j] + s;
					high_ij = high_ij - s;
					ret_row[j] = high_ij;

					// low_mat := mat - 2^shift * high_mat
					bnc_oz_mpf_set_scaled_d(tmp, high_ij, sigma);
					mpf_sub(tmp, get_mpfmatrix_ij(tmp_org_mat, i, j), tmp);
					set_mpfmatrix_ij(tmp_org_mat, i, j, tmp);
				}
			}

			mpf_clear(tmp);
		}

		real_num_div = index + 1;
	}

	free(power2);
	free(mantissa_buf);
	free(exponent_buf);
	free_mpfmatrix(tmp_org_mat);
	if(own_ret_mat != NULL)
		free_dmatrix(own_ret_mat);

	return real_num_div;
}

// SplitMat_A without the scaling; kept for callers that cannot apply a scale factor
int split_mpfmatrix_dmat(DMatrix ret_mat[], int num_div, MPFMatrix org_mat)
{
	return split_mpfmatrix_dmat_ex(ret_mat, NULL, num_div, org_mat);
}

// absmax_col_mpfmatrix
void absmax_col_mpfmatrix(mpf_t mu, long int *max_i, long int col_index, MPFMatrix mat)
{
    long int i, max_index = 0;
    mpf_t abs_aij;

    mpf_init2(abs_aij, mpf_get_prec(mu));

	//mu = fabs(mat[0 * col_dim + j]);
    mpf_abs(mu, get_mpfmatrix_ij(mat, 0, col_index));
	for(i = 1; i < mat->row_dim; i++)
	{
		mpf_abs(abs_aij, get_mpfmatrix_ij(mat, i, col_index));
		if(mpf_cmp(abs_aij, mu) > 0)
        {
			mpf_set(mu, abs_aij);
            max_index = i;
        }
	}

    if(max_i != NULL)
        *max_i = max_index;

    mpf_clear(abs_aij);

    return;
}

// SplitMat_B
/*------------------------------------------------------------------------------*/
/* SplitMat_B: same as split_mpfmatrix_dmat, but the threshold of the second     */
/* operand is taken over columns, so the maxima are reduced per column -- one    */
/* partial maximum vector per thread, combined afterwards, so that the sweep     */
/* can still walk the matrix row-wise.                                           */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
/* SplitMat_B: same as split_mpfmatrix_dmat_ex(), with the threshold and the     */
/* scale taken per column -- one partial maximum vector per thread, combined     */
/* afterwards, so that the sweep can still walk the matrix row-wise.             */
/*------------------------------------------------------------------------------*/
int split_mpfmatrix_t_dmat_ex(DMatrix ret_mat[], long int col_shift[], int num_div, MPFMatrix org_mat)
{
	unsigned long prec = org_mat->prec;
	long int i, j, index, row_dim, col_dim, ret_stride;
	int real_num_div, num_digits = 53, num_threads, thread; // IEEE double prec.
	double mu_total, tail_exp;
	MPFMatrix tmp_org_mat;
	DMatrix own_ret_mat = NULL, in_ret_mat;
	double *power2, *mu_local;
	long int *shift, *exp_local;

	row_dim = org_mat->row_dim;
	col_dim = org_mat->col_dim;

	if(ret_mat != NULL)
	{
		if((ret_mat[0]->row_dim != row_dim) || (ret_mat[0]->col_dim != col_dim))
		{
			fprintf(stderr, "ERROR: split_mpfmatrix_t_dmat\n");
			return 0;
		}
	}

	num_threads = bnc_oz_get_num_threads();

	power2 = (double *)calloc((size_t)col_dim, sizeof(double));
	mu_local = (double *)calloc((size_t)num_threads * (size_t)col_dim, sizeof(double));
	exp_local = (long int *)calloc((size_t)num_threads * (size_t)col_dim, sizeof(long int));
	if(mu_local == NULL || exp_local == NULL) // retry single-threaded rather than give up
	{
		num_threads = 1;
		free(mu_local);
		free(exp_local);
		mu_local = (double *)calloc((size_t)col_dim, sizeof(double));
		exp_local = (long int *)calloc((size_t)col_dim, sizeof(long int));
	}
	if(power2 == NULL || mu_local == NULL || exp_local == NULL)
	{
		fprintf(stderr, "ERROR: split_mpfmatrix_t_dmat: cannot allocate\n");
		free(power2);
		free(mu_local);
		free(exp_local);
		return 0;
	}

	if(ret_mat == NULL)
	{
		own_ret_mat = init_dmatrix(row_dim, col_dim);
		in_ret_mat = own_ret_mat;
	}
	else
		in_ret_mat = ret_mat[0];

	tmp_org_mat = init2_mpfmatrix(row_dim, col_dim, prec);
	subst_mpfmatrix(tmp_org_mat, org_mat);

	// the column-independent half of t_exp
	tail_exp = ceil(((double)num_digits + DLOG2((double)(row_dim))) / 2.0);

	real_num_div = 0;
	for(index = 0; index < num_div; index++)
	{
		if(ret_mat != NULL)
			in_ret_mat = ret_mat[index];
		ret_stride = in_ret_mat->real_col_dim;
		shift = (col_shift != NULL) ? (col_shift + (size_t)index * (size_t)col_dim) : NULL;

		// pass 1: the column scales, as exponents so that the mpf_t range is kept
		if(shift != NULL)
		{
			for(i = 0; i < (long int)num_threads * col_dim; i++)
				exp_local[i] = LONG_MIN;

#ifdef _OPENMP
			#pragma omp parallel num_threads(num_threads) private(i, j)
#endif // _OPENMP
			{
				long int *local_exp, element_exp;
#ifdef _OPENMP
				local_exp = exp_local + (size_t)omp_get_thread_num() * (size_t)col_dim;
				#pragma omp for schedule(static)
#else // _OPENMP
				local_exp = exp_local;
#endif // _OPENMP
				for(i = 0; i < row_dim; i++)
				{
					for(j = 0; j < col_dim; j++)
					{
						element_exp = bnc_oz_mpf_exp2(get_mpfmatrix_ij(tmp_org_mat, i, j));
						if(element_exp > local_exp[j])
							local_exp[j] = element_exp;
					}
				}
			}

			for(j = 0; j < col_dim; j++)
			{
				long int column_exp = exp_local[j];

				for(thread = 1; thread < num_threads; thread++)
				{
					if(exp_local[(size_t)thread * (size_t)col_dim + j] > column_exp)
						column_exp = exp_local[(size_t)thread * (size_t)col_dim + j];
				}

				shift[j] = (column_exp != LONG_MIN) ? column_exp : 0;
			}
		}

		// pass 2: the scaled double image and the column maxima
		for(i = 0; i < (long int)num_threads * col_dim; i++)
			mu_local[i] = 0.0;

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
				double *ret_row = in_ret_mat->element + i * ret_stride;
				double org_ij, abs_org_ij;

				for(j = 0; j < col_dim; j++)
				{
					org_ij = (shift != NULL && shift[j] != 0)
					         ? bnc_oz_mpf_get_scaled_d(get_mpfmatrix_ij(tmp_org_mat, i, j), shift[j])
					         : mpf_get_d(get_mpfmatrix_ij(tmp_org_mat, i, j));
					ret_row[j] = org_ij;

					abs_org_ij = fabs(org_ij);
					if(abs_org_ij > local_mu[j])
						local_mu[j] = abs_org_ij;
				}
			}
		}

		mu_total = 0.0;
		for(j = 0; j < col_dim; j++)
		{
			double mu = mu_local[j];

			for(thread = 1; thread < num_threads; thread++)
			{
				if(mu_local[(size_t)thread * (size_t)col_dim + j] > mu)
					mu = mu_local[(size_t)thread * (size_t)col_dim + j];
			}

			power2[j] = pow(2.0, ceil(DLOG2(mu)) + tail_exp);
			mu_total += mu;
		}

		if(mu_total == 0.0) break;

		// pass 3: high := (mat + s) - s, and mat := mat - 2^shift[j] * high
#ifdef _OPENMP
		#pragma omp parallel num_threads(num_threads) private(i, j)
#endif // _OPENMP
		{
			mpf_t tmp;

			mpf_init2(tmp, prec);

#ifdef _OPENMP
			#pragma omp for schedule(static)
#endif // _OPENMP
			for(i = 0; i < row_dim; i++)
			{
				double *ret_row = in_ret_mat->element + i * ret_stride;
				double s, high_ij;

				for(j = 0; j < col_dim; j++)
				{
					s = power2[j];
					high_ij = ret_row[j] + s;
					high_ij = high_ij - s;
					ret_row[j] = high_ij;

					bnc_oz_mpf_set_scaled_d(tmp, high_ij, (shift != NULL) ? shift[j] : 0);
					mpf_sub(tmp, get_mpfmatrix_ij(tmp_org_mat, i, j), tmp);
					set_mpfmatrix_ij(tmp_org_mat, i, j, tmp);
				}
			}

			mpf_clear(tmp);
		}

		real_num_div = index + 1;
	}

	free(power2);
	free(mu_local);
	free(exp_local);
	free_mpfmatrix(tmp_org_mat);
	if(own_ret_mat != NULL)
		free_dmatrix(own_ret_mat);

	return real_num_div;
}

// SplitMat_B without the scaling; kept for callers that cannot apply a scale factor
int split_mpfmatrix_t_dmat(DMatrix ret_mat[], int num_div, MPFMatrix org_mat)
{
	return split_mpfmatrix_t_dmat_ex(ret_mat, NULL, num_div, org_mat);
}

// Matrix multiplication based on Ozaki scheme
/*------------------------------------------------------------------------------*/
/* Matrix multiplication based on Ozaki scheme                                   */
/*                                                                               */
/* ret = sum_{i,j} div_a[i] * div_b[j].  In BNC_OZ_GEMM_MODE_OWN the rows of ret */
/* are cut into blocks and one thread takes a block at a time, running every     */
/* slice product for it with a single-threaded DGEMM and accumulating in mpf_t   */
/* on the spot; the accumulation, which a threaded BLAS would leave serial, is   */
/* by far the most expensive part here, so parallelizing it matters more than    */
/* the products themselves.  Blocks are disjoint and each element of ret still   */
/* sums its slice products in the original order, so the result is identical to  */
/* the serial one.                                                               */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
/* Matrix multiplication based on Ozaki scheme                                   */
/*                                                                               */
/* ret = sum_{i,j} 2^(sa[i] + sb[j]) div_a[i] * div_b[j].  In                    */
/* BNC_OZ_GEMM_MODE_OWN the rows of ret are cut into blocks and one thread takes */
/* a block at a time, running every slice product for it with a single-threaded  */
/* DGEMM and accumulating in mpf_t on the spot; the accumulation, which a        */
/* threaded BLAS would leave serial, is by far the most expensive part here, so  */
/* parallelizing it matters more than the products themselves.  Blocks are       */
/* disjoint and each element of ret still sums its slice products in the         */
/* original order, so the result does not depend on the number of threads.       */
/*                                                                               */
/* The two exponents the slices were scaled by are put back on the product here, */
/* which is what lets the operands use the whole mpf_t exponent range.           */
/*------------------------------------------------------------------------------*/
void mul_mpfmatrix_oz(MPFMatrix ret, MPFMatrix a, int max_num_div_a, MPFMatrix b, int max_num_div_b)
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
        fprintf(stderr, "ERROR: mul_mpfmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
        return;
    }

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_b = (DMatrix *)calloc(max_num_div_b, sizeof(DMatrix));
    row_shift = (long int *)calloc((size_t)max_num_div_a * (size_t)row_dim, sizeof(long int));
    col_shift = (long int *)calloc((size_t)max_num_div_b * (size_t)col_dim, sizeof(long int));
    if(div_a == NULL || div_b == NULL || row_shift == NULL || col_shift == NULL)
    {
        fprintf(stderr, "ERROR: mul_mpfmatrix_oz: cannot allocate\n");
        free(div_a); free(div_b); free(row_shift); free(col_shift);
        return;
    }
    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, mid_dim);
    for(i = 0; i < max_num_div_b; i++)
        div_b[i] = init_dmatrix(mid_dim, col_dim);

    real_num_div_a = split_mpfmatrix_dmat_ex(div_a, row_shift, max_num_div_a, a);
    real_num_div_b = split_mpfmatrix_t_dmat_ex(div_b, col_shift, max_num_div_b, b);

    set0_mpfmatrix(ret);

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
            int div_i, div_j;
            double *buf;
            mpf_t tmp;

            mpf_init2(tmp, ret->prec);

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

                            for(jj = 0; jj < col_dim; jj++)
                            {
                                if(buf_row[jj] == 0.0)
                                    continue;

                                bnc_oz_mpf_set_scaled_d(tmp, buf_row[jj], shift_a + shift_b[jj]);
                                mpf_add(tmp, get_mpfmatrix_ij(ret, row_start + ii, jj), tmp);
                                set_mpfmatrix_ij(ret, row_start + ii, jj, tmp);
                            }
                        }
                    }
                }
            }

            mpf_clear(tmp);
        }

        bnc_oz_blas_leave(prev_blas_threads);
        free(block_buf);
    }
    else // one full-size product per slice pair, threaded by the BLAS if it can
    {
        long int ii, jj;
        mpf_t tmp;

        mpf_init2(tmp, ret->prec);
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

                    for(jj = 0; jj < col_dim; jj++)
                    {
                        if(buf_row[jj] == 0.0)
                            continue;

                        bnc_oz_mpf_set_scaled_d(tmp, buf_row[jj], shift_a + shift_b[jj]);
                        mpf_add(tmp, get_mpfmatrix_ij(ret, ii, jj), tmp);
                        set_mpfmatrix_ij(ret, ii, jj, tmp);
                    }
                }
            }
        }

        mpf_clear(tmp);
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
/* Same blocking as mul_mpfmatrix_oz: a thread owns a range of rows of ret and   */
/* runs all slice pairs for it, so the accumulation is parallel too.             */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
/* Matrix-Vector multiplication based on Ozaki scheme                            */
/*                                                                               */
/* Same blocking and the same exponent handling as mul_mpfmatrix_oz(): a thread  */
/* owns a range of rows of ret and runs all slice pairs for it.                  */
/*------------------------------------------------------------------------------*/
void mul_mpfmatrix_mpfvec_oz(MPFVector ret, MPFMatrix a, int max_num_div_a, MPFVector vb, int max_num_div_vb)
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
        fprintf(stderr, "ERROR: mul_mpfmatrix_mpfvec_oz: cannot allocate\n");
        free(div_a); free(div_vb); free(row_shift); free(vec_shift);
        return;
    }

    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, col_dim);
    for(i = 0; i < max_num_div_vb; i++)
        div_vb[i] = init_dvector(vb->dim); // vb->dim, not ret->dim: they differ when a is not square

    real_num_div_a = split_mpfmatrix_dmat_ex(div_a, row_shift, max_num_div_a, a);
    real_num_div_vb = split_mpfvector_dvec_ex(div_vb, vec_shift, max_num_div_vb, vb);

    set0_mpfvector(ret);

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
            int div_i, div_j;
            double *buf;
            mpf_t tmp;

            mpf_init2(tmp, ret->prec);

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
                            if(buf[ii] == 0.0)
                                continue;

                            shift_a = row_shift[(size_t)div_i * (size_t)row_dim + row_start + ii];

                            bnc_oz_mpf_set_scaled_d(tmp, buf[ii], shift_a + vec_shift[div_j]);
                            mpf_add(tmp, get_mpfvector_i(ret, row_start + ii), tmp);
                            set_mpfvector_i(ret, row_start + ii, tmp);
                        }
                    }
                }
            }

            mpf_clear(tmp);
        }

        bnc_oz_blas_leave(prev_blas_threads);
        free(block_buf);
    }
    else
    {
        long int ii;
        mpf_t tmp;

        mpf_init2(tmp, ret->prec);
        div_ret = init_dvector(vec_dim);

        for(i = 0; i < real_num_div_a; i++)
        {
            for(j = 0; j < real_num_div_vb; j++)
            {
                bnc_oz_dgemv_block(div_ret->element, div_a[i], 0, vec_dim, div_vb[j]);

                for(ii = 0; ii < vec_dim; ii++)
                {
                    long int shift_a = row_shift[(size_t)i * (size_t)row_dim + ii];

                    if(div_ret->element[ii] == 0.0)
                        continue;

                    bnc_oz_mpf_set_scaled_d(tmp, div_ret->element[ii], shift_a + vec_shift[j]);
                    mpf_add(tmp, get_mpfvector_i(ret, ii), tmp);
                    set_mpfvector_i(ret, ii, tmp);
                }
            }
        }

        mpf_clear(tmp);
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

// Matrix multiplication based on Ozaki scheme (4M)
void mul_cmpfmatrix_oz_4m(CMPFMatrix ret, CMPFMatrix a, int max_num_div_a_real, int max_num_div_a_image, CMPFMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    MPFMatrix a_real, a_image, b_real, b_image, c_real[2], c_image[2];

    a_real  = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    a_image = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    b_real  = init2_mpfmatrix(b->row_dim, b->col_dim, b->prec);
    b_image = init2_mpfmatrix(b->row_dim, b->col_dim, b->prec);
    c_real[0]  = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    c_image[0] = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    c_real[1]  = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    c_image[1] = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);

    // A_r + A_i * i := A
    separate_cmpfmatrix(a_real, a_image, a);
    separate_cmpfmatrix(b_real, b_image, b);

    // C := (A_r + A_i *i) * (B_r + B_i * i)
    //    = (A_r + B_r - A_i * B_i) + (A_r * B_i + A_i * B_r) * i
    mul_mpfmatrix_oz(c_real[0], a_real, max_num_div_a_real, b_real, max_num_div_b_real);
    mul_mpfmatrix_oz(c_real[1], a_image, max_num_div_a_image, b_image, max_num_div_b_image);
    sub_mpfmatrix(c_real[0], c_real[0], c_real[1]);

    mul_mpfmatrix_oz(c_image[0], a_real, max_num_div_a_real, b_image, max_num_div_b_image);
    mul_mpfmatrix_oz(c_image[1], a_image, max_num_div_a_image, b_real, max_num_div_b_real);
    add_mpfmatrix(c_image[0], c_image[0], c_image[1]);

    // C := C_r + C_i * i
    merge_cmpfmatrix(ret, c_real[0], c_image[0]);

    free_mpfmatrix(a_real);
    free_mpfmatrix(a_image);
    free_mpfmatrix(b_real);
    free_mpfmatrix(b_image);
    free_mpfmatrix(c_real[0]);
    free_mpfmatrix(c_image[0]);
    free_mpfmatrix(c_real[1]);
    free_mpfmatrix(c_image[1]);
}

// Matrix multiplication based on Ozaki scheme (3M)
void mul_cmpfmatrix_oz_3m(CMPFMatrix ret, CMPFMatrix a, int max_num_div_a_real, int max_num_div_a_image, CMPFMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    int max_num_div_apa, max_num_div_bpb;
    MPFMatrix a_real, a_image, b_real, b_image, c_real, c_image, apa, bpb, t[2];

    // Allocate
    a_real  = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    a_image = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    apa     = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);

    b_real  = init2_mpfmatrix(b->row_dim, b->col_dim, b->prec);
    b_image = init2_mpfmatrix(b->row_dim, b->col_dim, b->prec);
    bpb     = init2_mpfmatrix(b->row_dim, b->col_dim, b->prec);

    c_real  = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    c_image = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    t[0] = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    t[1] = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);

    // A_r + A_i * i := A
    // B_r + B_i * i := B
    separate_cmpfmatrix(a_real, a_image, a);
    separate_cmpfmatrix(b_real, b_image, b);

    // T0 := A_r * B_r
    // T1 := A_i * B_i
    mul_mpfmatrix_oz(t[0], a_real, max_num_div_a_real, b_real, max_num_div_b_real);
    mul_mpfmatrix_oz(t[1], a_image, max_num_div_a_image, b_image, max_num_div_b_image);

    // C_r := T0 - T1
    sub_mpfmatrix(c_real, t[0], t[1]);

    // C_i := (A_r + A_i) * (B_r + B_i) - T0 - T1
    add_mpfmatrix(apa, a_real, a_image);
    add_mpfmatrix(bpb, b_real, b_image);
    max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
    max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
    mul_mpfmatrix_oz(c_image, apa, max_num_div_apa, bpb, max_num_div_bpb);
    sub_mpfmatrix(c_image, c_image, t[0]);
    sub_mpfmatrix(c_image, c_image, t[1]);

    // C := C_r + C_i * i
    merge_cmpfmatrix(ret, c_real, c_image);

    // Free
    free_mpfmatrix(a_real);
    free_mpfmatrix(a_image);
    free_mpfmatrix(apa);
    free_mpfmatrix(b_real);
    free_mpfmatrix(b_image);
    free_mpfmatrix(bpb);
    free_mpfmatrix(c_real);
    free_mpfmatrix(c_image);
    free_mpfmatrix(t[0]);
    free_mpfmatrix(t[1]);

}
//#endif // not DEBUG

// Testing
#ifdef DEBUG

#include "get_secv.h"

//#define dim 5
//#define DIM 10
#define DIM 128
//#define DIM 512

//#define MAX_NUM_DIV 10
//#define MAX_NUM_DIV 15
#define MAX_NUM_DIV 128

int main(int argc, char *argv[])
{
    unsigned long prec;
    long int i, j, dim;
    int real_num_div, num_div;
    mpf_t tmp, tmp2;
    MPFVector vec_org, vec;
    DVector dvec[MAX_NUM_DIV];
    MPFMatrix mat_a, mat_b, mat_c, mat;
    DMatrix dmat[MAX_NUM_DIV];
    mpf_t ddtmp, ddtmp2;
    double stime, etime;

    //goto mpfmatrix;
    if(argc <= 2)
    {
        printf("$ %s prec [dim] [num_div]\n", argv[0]);
        return 0;
    }
    prec = atoi(argv[1]);
    set_bnc_default_prec(prec);

    mpf_init(tmp); mpf_init(tmp2);
    mpf_init(ddtmp); mpf_init(ddtmp2);

    dim = DIM;
    if(argc >= 3)
        dim = atoi(argv[2]);

    num_div = MAX_NUM_DIV;
    if(argc >= 4)
        num_div = atoi(argv[3]);

    //goto ddmatrix;

// DVector 
    vec_org = init_mpfvector(dim);
    vec = init_mpfvector(dim);
    for(i = 0; i < num_div; i++)
        dvec[i] = init_dvector(dim);

    // set random
    srand(dim);
    for(i = 0; i < dim; i++)
    {
        mpf_set_d(tmp, (double)rand());
        mpf_mul_d(tmp, tmp, (double)rand());
        mpf_div_d(tmp, tmp, (double)rand());
        set_mpfvector_i(vec_org, i, tmp);
    }

    // split
    real_num_div = split_mpfvector_dvec(dvec, num_div, vec_org); //, 53);

    for(i = 0; i < num_div; i++)
    {
        //printf("dvec[%d]:\n", i); print_dvector(dvec[i]);

        add_mpfvector_dvec(vec, vec, dvec[i]);
    }
    //sub_mpfvector(vec, vec, vec_org);
    //printf("vec - vec_org:\n"); print_mpfvector(vec);
    //printf("\n");
    /*for(i = 0; i < dim; i++)
    {
        ddtmp = get_mpfvector_i_tdfloat(vec, i);
        ddtmp2 = get_mpfvector_i_tdfloat(vec_org, i);
        printf("%5d %25.17e %25.17e %25.17e\n%5d %25.17e %25.17e %25.17e\n", i, ddtmp.val[0], ddtmp.val[1], ddtmp.val[2], i, ddtmp2.val[0], ddtmp2.val[1], ddtmp.val[2]);
    }
    */
    

    free_mpfvector(vec_org);
    free_mpfvector(vec);
    for(i = 0; i < 10; i++)
        free_dvector(dvec[i]);

    //sgoto end;
// DMatrix
mpfmatrix:

    mat_a = init_mpfmatrix(dim, dim);
    mat_b = init_mpfmatrix(dim, dim);
    mat_c = init_mpfmatrix(dim, dim);
    mat = init_mpfmatrix(dim, dim);
    for(i = 0; i < num_div; i++)
        dmat[i] = init_dmatrix(dim, dim);

    // set random
    srand(dim);
    for(i = 0; i < dim; i++)
    {
        for(j = 0; j < dim; j++)
        {
            mpf_set_d(tmp, (double)rand() / (double)RAND_MAX);
            mpf_mul_d(tmp, tmp, (double)rand());
            mpf_div_d(tmp, tmp, (double)rand());
            set_mpfmatrix_ij(mat_a, i, j, tmp);

            mpf_set_d(tmp, (double)rand() / (double)RAND_MAX);
            mpf_mul_d(tmp, tmp, (double)rand());
            mpf_div_d(tmp, tmp, (double)rand());
            set_mpfmatrix_ij(mat_b, i, j, tmp);
        }
    }

    // split_A
    printf("Split_A:\n");
    real_num_div = split_mpfmatrix_dmat(dmat, num_div, mat_a);
    printf("num_div, real_num_div = %d, %d\n", num_div, real_num_div);

    for(i = 0; i < real_num_div; i++)
    {
        //printf("dmat[%d]:\n", i); print_dmatrix(dmat[i]);

        add_mpfmatrix_dmat(mat, mat, dmat[i]);
    }
    sub_mpfmatrix(mat, mat, mat_a);
    printf("||mat - mat_org|| / ||mat_org||:\n"); 
    normf_mpfmatrix(tmp, mat);
    normf_mpfmatrix(tmp2, mat_a);
    mpf_div(tmp, tmp, tmp2);
    mpf_out_str(stdout, 10, 0, tmp); //print_mpfmatrix(mat);
    printf("\n"); 

    // split_B
    printf("Split_B:\n");
    real_num_div = split_mpfmatrix_t_dmat(dmat, num_div, mat_b);
    printf("num_div, real_num_div = %d, %d\n", num_div, real_num_div);

    for(i = 0; i < real_num_div; i++)
    {
        //printf("dmat[%d]:\n", i); print_dmatrix(dmat[i]);

        add_mpfmatrix_dmat(mat, mat, dmat[i]);
    }
    sub_mpfmatrix(mat, mat, mat_b);
    printf("||mat - mat_org|| / ||mat_org||:\n"); 
    normf_mpfmatrix(tmp, mat);
    normf_mpfmatrix(tmp2, mat_a);
    mpf_div(tmp, tmp, tmp2);
    mpf_out_str(stdout, 10, 0, tmp); //print_mpfmatrix(mat);
    printf("\n"); 

    // a * b
    stime = get_secv();
    mul_mpfmatrix_oz(mat, mat_a, num_div, mat_b, num_div);
    etime = get_secv() - stime;
    printf("mul_mpfmatrix_oz(dim, prec, sec)      : %5d, %5d, %10.3g\n", dim, prec, etime);

    stime = get_secv();
    //mul_mpfmatrix(mat_c, mat_a, mat_b);
    mul_mpfmatrix_strassen(mat_c, mat_a, mat_b, 32);
    etime = get_secv() - stime;
    printf("mul_mpfmatrix_strassen(dim, prec, sec): %5d, %5d, %10.3g\n", dim, prec, etime);

    sub_mpfmatrix(mat, mat, mat_c);
    printf("||mat_oz - mat_org|| / ||mat_org||:\n"); 
    normf_mpfmatrix(tmp, mat);
    normf_mpfmatrix(tmp2, mat_c);
    mpf_div(tmp, tmp, tmp2);
    mpf_out_str(stdout, 10, 0, tmp); //print_mpfmatrix(mat); 
    printf("\n"); 

    free_mpfmatrix(mat_a);
    free_mpfmatrix(mat_b);
    free_mpfmatrix(mat_c);
    free_mpfmatrix(mat);
    for(i = 0; i < 10; i++)
        free_dmatrix(dmat[i]);

    mpf_clear(tmp); mpf_clear(tmp2);
    mpf_clear(ddtmp); mpf_clear(ddtmp2);

end:
    return 0;
}

// Testing
#endif // DEBUG
