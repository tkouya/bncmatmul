/********************************************************************************/
/* qd_oz_scheme: Multiple precision linear computation based on Ozaki scheme.   */
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

// log2(x) := log10(x) / log10(2)
//#define DLOG2(x) (log10((x)) / 0.30102999566398119521373889472449)

// absmax_qdvector
void absmax_qdvector(double ret[QDSIZE], long int *max_index, QDVector vec)
{
    long int i, max_i, dim = vec->dim;
    double abs_val[QDSIZE];

    max_i = 0;
    set0_qd(ret);
    for(i = 0; i < dim; i++)
    {
        //abs_val = fabs(get_qdvector_i(vec, i));
        rqd_abs(abs_val, get_qdvector_i(vec, i));
        if(rqd_cmp(ret, abs_val) < 0)
        {
            rqd_set(ret, abs_val);
            max_i = i;
        }
    }

    if(max_index != NULL)
        *max_index = max_i;

    return;
}

/* c = a + (double)b */
void add_qdvector_dvec(QDVector c, QDVector a, DVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_qdvector_dvec\n");
		return;
	}

	double tmp[QDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rqd_add_d(tmp, get_qdvector_i(a, i), get_dvector_i(b, i));
		set_qdvector_i(c, i, tmp);
	}
}

/* c = a - (double)b */
void sub_qdvector_dvec(QDVector c, QDVector a, DVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_qdvector_dvec\n");
		return;
	}

	double tmp[QDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rqd_sub_d(tmp, get_qdvector_i(a, i),  get_dvector_i(b, i));
		set_qdvector_i(c, i, tmp);
	}
}

/* c := (d)a */
void subst_qdvector_dvec(QDVector c, DVector a)
{
	long int i, j;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: subst_qdvector_dvec\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
		set_qdvector_i_d(c, i, get_dvector_i(a, i));
}

/* c := (qd)a */
void subst_dvector_qdvec(DVector c, QDVector a)
{
	long int i, j;
    qdfloat tmp;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: subst_dvector_qdvec\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
	{
        tmp = get_qdvector_i_qdfloat(a, i);
		set_dvector_i(c, i, tmp.val[0]);
	}
}
// split vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
//void extract_qdvector(DVector ret_vec[], int num_div, QDVector org_vec, int num_bits)
/*------------------------------------------------------------------------------*/
/* split vector: 2^shift[index] * (ret_vec[0] + ... ) = org_vec                  */
/*                                                                               */
/* The vector plays the role of a single column of B, so it carries one exponent */
/* per slice.  shift may be NULL; see oz_scheme.h.                               */
/*------------------------------------------------------------------------------*/
int split_qdvector_dvec_ex(DVector ret_vec[], long int shift[], int num_div, QDVector org_vec)
{
	long int dim = org_vec->dim, i;
	int index, num_bits = 53, real_num_div; // IEEE754 binary64
	double absmax_org_vec, threshold, t_exp, tail_exp, org_vec_i, high_i;
	double org_ii[QDSIZE], high_qd[QDSIZE], rest_i[QDSIZE];
	long int sigma;
	QDVector tmp_org_vec;
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
	tmp_org_vec = init_qdvector(dim);
	subst_qdvector(tmp_org_vec, org_vec);

	tail_exp = ceil(((double)num_bits + DLOG2((double)(dim))) / 2.0);

	for(comp = 0; comp < QDSIZE; comp++)
		high_qd[comp] = 0.0;

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

			for(comp = 0; comp < QDSIZE; comp++)
				org_ii[comp] = tmp_org_vec->element[comp][i];
			high_qd[0] = (sigma != 0) ? bnc_oz_ldexp(high_i, sigma) : high_i;

			rqd_sub(rest_i, org_ii, high_qd);

			for(comp = 0; comp < QDSIZE; comp++)
				tmp_org_vec->element[comp][i] = rest_i[comp];
		}

		real_num_div = index + 1;
	}

	free_qdvector(tmp_org_vec);
	if(own_ret_vec != NULL)
		free_dvector(own_ret_vec);

	return real_num_div;
}

// split vector without the scaling
int split_qdvector_dvec(DVector ret_vec[], int num_div, QDVector org_vec)
{
	return split_qdvector_dvec_ex(ret_vec, NULL, num_div, org_vec);
}

// absmax_row_qdmatrix
void absmax_row_qdmatrix(double mu[QDSIZE], long int *max_j, long int row_index, QDMatrix mat)
{
    long int j, max_index = 0;
    double abs_aij[QDSIZE];

	//mu = fabs(mat[i * col_dim + 0]);
    rqd_abs(mu, get_qdmatrix_ij(mat, row_index, 0));

	for(j = 1; j < mat->col_dim; j++)
	{
		rqd_abs(abs_aij, get_qdmatrix_ij(mat, row_index, j));
		if(rqd_cmp(abs_aij, mu) > 0)
        {
			//mu = abs_aij;
            rqd_set(mu, abs_aij);
            max_index = j;
        }
	}

    if(max_j != NULL)
        *max_j = max_index;

    //return mu;
    return;
}

/* c := a + (doble)b */
void add_qdmatrix_dmat(QDMatrix c, QDMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim, a_stride, b_stride, c_stride;
	int num_threads;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_qdmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_qdmatrix_dmat\n");
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
		const double *a_row[QDSIZE];
		const double *b_row = b->element + i * b_stride;
		double *c_row[QDSIZE];
		double a_ij[QDSIZE], b_ij[QDSIZE], c_ij[QDSIZE];
		int comp;

		for(comp = 0; comp < QDSIZE; comp++)
		{
			a_row[comp] = a->element[comp] + i * a_stride;
			c_row[comp] = c->element[comp] + i * c_stride;
			b_ij[comp] = 0.0;
		}

		for(j = 0; j < col_dim; j++)
		{
			for(comp = 0; comp < QDSIZE; comp++)
				a_ij[comp] = a_row[comp][j];
			b_ij[0] = b_row[j];

			rqd_add(c_ij, a_ij, b_ij);

			for(comp = 0; comp < QDSIZE; comp++)
				c_row[comp][j] = c_ij[comp];
		}
	}
}

/* c := a - (doble)b */
void sub_qdmatrix_dmat(QDMatrix c, QDMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim, a_stride, b_stride, c_stride;
	int num_threads;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_qdmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_qdmatrix_dmat\n");
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
		const double *a_row[QDSIZE];
		const double *b_row = b->element + i * b_stride;
		double *c_row[QDSIZE];
		double a_ij[QDSIZE], b_ij[QDSIZE], c_ij[QDSIZE];
		int comp;

		for(comp = 0; comp < QDSIZE; comp++)
		{
			a_row[comp] = a->element[comp] + i * a_stride;
			c_row[comp] = c->element[comp] + i * c_stride;
			b_ij[comp] = 0.0;
		}

		for(j = 0; j < col_dim; j++)
		{
			for(comp = 0; comp < QDSIZE; comp++)
				a_ij[comp] = a_row[comp][j];
			b_ij[0] = b_row[j];

			rqd_sub(c_ij, a_ij, b_ij);

			for(comp = 0; comp < QDSIZE; comp++)
				c_row[comp][j] = c_ij[comp];
		}
	}
}

/* c := (d)a */
void subst_qdmatrix_dmat(QDMatrix c, DMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qdmatrix_dmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_qdmatrix_ij_d(c, i, j, get_dmatrix_ij(a, i, j));
		}
	}
}

/* c := (qd)a */
void subst_dmatrix_qdmat(DMatrix c, QDMatrix a)
{
	long int i, j, real_row_dim, real_col_dim, real_total_dim;
    qdfloat tmp;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_dmatrix_qdmat\n");
		return;
	}

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
            tmp = get_qdmatrix_ij_qdfloat(a, i, j);
			set_dmatrix_ij(c, i, j, tmp.val[0]);
		}
	}
/*	__m256d aij4;;

    real_row_dim = c->real_row_dim;
    real_col_dim = c->real_col_dim;
    real_total_dim = real_row_dim * real_col_dim;
    memcpy((void *)(c->element), (void *)(a->element[0]), (size_t)(real_total_dim * sizeof(double)));
*/
    /*
    for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
    {
        aij4 = _mm256_load_pd(&(a->element[0][i]));
        //_mm256_store_pd(&(get_dmatrix_ij(c, (i / real_col_dim), (i % real_col_dim))), aij4);
        _mm256_store_pd(&(c->element[i]), aij4);
    }
    */

#elif defined(__AVX512F__) // __AVX512F__
	//__m512d tmp8[QDSIZE], aij8[QDSIZE], bij8;

    real_total_dim = c->real_row_dim * c->real_col_dim;
    //memcpy(c->element, a->element[0], real_total_dim * sizeof(double));

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, hi-limb copy)
    {
        long int index;
        real_total_dim = c->real_row_dim * c->real_col_dim;
        for(index = 0; index < real_total_dim; index += (long int)svcntd())
        {
            svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)real_total_dim);
            svst1_f64(pg, &(c->element[index]), svld1_f64(pg, &(a->element[0][index])));
        }
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon (hi-limb copy)
    {
        long int index;
        real_total_dim = c->real_row_dim * c->real_col_dim;
        for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
            vst1q_f64(&(c->element[index]), vld1q_f64(&(a->element[0][index])));
    }
#else // __AVX2__
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
            tmp = get_qdmatrix_ij_qdfloat(a, i, j);
			set_dmatrix_ij(c, i, j, tmp.val[0]);
		}
	}
#endif // __AVX2__
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
int split_qdmatrix_dmat_ex(DMatrix ret_mat[], long int row_shift[], int num_div, QDMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, org_stride, ret_stride;
	long int num_digits = 53; // IEEE double prec.
	int real_num_div, num_threads;
	double mu_total, tail_exp;
	QDMatrix tmp_org_mat;
	DMatrix own_ret_mat = NULL, in_ret_mat;
	double *power2; // 2^t_exp of each row, in the scaled domain
	long int *shift;

	row_dim = org_mat->row_dim;
	col_dim = org_mat->col_dim;

	if(ret_mat != NULL)
	{
		if((ret_mat[0]->row_dim != row_dim) || (ret_mat[0]->col_dim != col_dim))
		{
			fprintf(stderr, "ERROR: split_qdmatrix_dmat\n");
			return 0;
		}
	}

	power2 = (double *)calloc((size_t)row_dim, sizeof(double));
	if(power2 == NULL)
	{
		fprintf(stderr, "ERROR: split_qdmatrix_dmat: cannot allocate\n");
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
	tmp_org_mat = init_qdmatrix(row_dim, col_dim);
	subst_qdmatrix(tmp_org_mat, org_mat);
	org_stride = tmp_org_mat->real_col_dim;

	// the row-independent half of t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim) / 2)
	tail_exp = ceil(((double)num_digits + DLOG2((double)col_dim)) / 2.0);

	num_threads = bnc_oz_get_num_threads();

	real_num_div = 0;
	for(index = 0; index < num_div - 1; index++)
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
				abs_org_ij = fabs(org_row[j]); // == |rqd_get_d(org[i][j])|
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
			double *org_row[QDSIZE];
			double s = power2[i], high_ij;
			double org_ij[QDSIZE], high_qd[QDSIZE], rest_ij[QDSIZE];
			long int sigma = (shift != NULL) ? shift[i] : 0;
			int comp;

			for(comp = 0; comp < QDSIZE; comp++)
			{
				org_row[comp] = tmp_org_mat->element[comp] + i * org_stride;
				high_qd[comp] = 0.0;
			}

			for(j = 0; j < col_dim; j++)
			{
				// (x + s) - s keeps the leading bits of x; valid under the IEEE
				// semantics this library is compiled with (no -ffast-math)
				high_ij = ret_row[j] + s;
				high_ij = high_ij - s;
				ret_row[j] = high_ij;

				// low_mat := mat - 2^shift * high_mat
				for(comp = 0; comp < QDSIZE; comp++)
					org_ij[comp] = org_row[comp][j];
				high_qd[0] = (sigma != 0) ? bnc_oz_ldexp(high_ij, sigma) : high_ij;

				rqd_sub(rest_ij, org_ij, high_qd);

				for(comp = 0; comp < QDSIZE; comp++)
					org_row[comp][j] = rest_ij[comp];
			}
		}

		real_num_div = index + 1;
	}

	free(power2);
	free_qdmatrix(tmp_org_mat);
	if(own_ret_mat != NULL)
		free_dmatrix(own_ret_mat);

	return real_num_div;
}

// SplitMat_A without the scaling; kept for callers that cannot apply a scale factor
int split_qdmatrix_dmat(DMatrix ret_mat[], int num_div, QDMatrix org_mat)
{
	return split_qdmatrix_dmat_ex(ret_mat, NULL, num_div, org_mat);
}

// absmax_col_qdmatrix
void absmax_col_qdmatrix(double mu[QDSIZE], long int *max_i, long int col_index, QDMatrix mat)
{
    long int i, max_index = 0;
    double abs_aij[QDSIZE];

	//mu = fabs(mat[0 * col_dim + j]);
    rqd_abs(mu, get_qdmatrix_ij(mat, 0, col_index));
	for(i = 1; i < mat->row_dim; i++)
	{
		rqd_abs(abs_aij, get_qdmatrix_ij(mat, i, col_index));
		if(rqd_cmp(abs_aij, mu) > 0)
        {
			rqd_set(mu, abs_aij);
            max_index = i;
        }
	}

    if(max_i != NULL)
        *max_i = max_index;

    return;
}

// SplitMat_B
/*------------------------------------------------------------------------------*/
/* SplitMat_B: same as split_qdmatrix_dmat, but the threshold of the second   */
/* operand is taken over columns, so the maxima are reduced per column.  The     */
/* sweep still walks the matrix row-wise -- one partial maximum vector per       */
/* thread, combined afterwards -- because the column-wise walk the definition    */
/* suggests would touch a new cache line on every element.                       */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
/* SplitMat_B: same as split_qdmatrix_dmat_ex(), but the threshold and the    */
/* scale of the second operand are taken over columns, so the maxima are reduced */
/* per column -- one partial maximum vector per thread, combined afterwards, so  */
/* that the sweep can still walk the matrix row-wise.                            */
/*------------------------------------------------------------------------------*/
int split_qdmatrix_t_dmat_ex(DMatrix ret_mat[], long int col_shift[], int num_div, QDMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, org_stride, ret_stride;
	int real_num_div, num_digits = 53, num_threads, thread; // IEEE double prec.
	double mu_total, tail_exp;
	QDMatrix tmp_org_mat;
	DMatrix own_ret_mat = NULL, in_ret_mat;
	double *power2, *mu_local; // 2^t_exp of each column, per-thread column maxima
	long int *shift;

	row_dim = org_mat->row_dim;
	col_dim = org_mat->col_dim;

	if(ret_mat != NULL)
	{
		if((ret_mat[0]->row_dim != row_dim) || (ret_mat[0]->col_dim != col_dim))
		{
			fprintf(stderr, "ERROR: split_qdmatrix_t_dmat\n");
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
		fprintf(stderr, "ERROR: split_qdmatrix_t_dmat: cannot allocate\n");
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

	tmp_org_mat = init_qdmatrix(row_dim, col_dim);
	subst_qdmatrix(tmp_org_mat, org_mat);
	org_stride = tmp_org_mat->real_col_dim;

	// the column-independent half of t_exp
	tail_exp = ceil(((double)num_digits + DLOG2((double)(row_dim))) / 2.0);

	real_num_div = 0;
	for(index = 0; index < num_div - 1; index++)
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
			double *org_row[QDSIZE];
			double s, high_ij, scaled_ij;
			double org_ij[QDSIZE], high_qd[QDSIZE], rest_ij[QDSIZE];
			long int sigma;
			int comp;

			for(comp = 0; comp < QDSIZE; comp++)
			{
				org_row[comp] = tmp_org_mat->element[comp] + i * org_stride;
				high_qd[comp] = 0.0;
			}

			for(j = 0; j < col_dim; j++)
			{
				sigma = (shift != NULL) ? shift[j] : 0;
				s = power2[j];

				scaled_ij = (sigma != 0) ? bnc_oz_ldexp(org_row[0][j], -sigma) : org_row[0][j];

				high_ij = scaled_ij + s;
				high_ij = high_ij - s;
				ret_row[j] = high_ij;

				for(comp = 0; comp < QDSIZE; comp++)
					org_ij[comp] = org_row[comp][j];
				high_qd[0] = (sigma != 0) ? bnc_oz_ldexp(high_ij, sigma) : high_ij;

				rqd_sub(rest_ij, org_ij, high_qd);

				for(comp = 0; comp < QDSIZE; comp++)
					org_row[comp][j] = rest_ij[comp];
			}
		}

		real_num_div = index + 1;
	}

	free(power2);
	free(mu_local);
	free_qdmatrix(tmp_org_mat);
	if(own_ret_mat != NULL)
		free_dmatrix(own_ret_mat);

	return real_num_div;
}

// SplitMat_B without the scaling; kept for callers that cannot apply a scale factor
int split_qdmatrix_t_dmat(DMatrix ret_mat[], int num_div, QDMatrix org_mat)
{
	return split_qdmatrix_t_dmat_ex(ret_mat, NULL, num_div, org_mat);
}

// Matrix multiplication based on Ozaki scheme
/*------------------------------------------------------------------------------*/
/* Matrix multiplication based on Ozaki scheme                                   */
/*                                                                               */
/* ret = sum_{i,j} div_a[i] * div_b[j].  In BNC_OZ_GEMM_MODE_OWN the rows of ret */
/* are cut into blocks and one thread takes a block at a time, running every     */
/* slice product for it with a single-threaded DGEMM and accumulating in QD   */
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
/* DGEMM and accumulating in QD on the spot; the accumulation, which a        */
/* threaded BLAS would leave serial, is thereby parallelized as well and the     */
/* block of ret stays hot in cache across all the slice pairs.  Blocks are       */
/* disjoint and each element of ret still sums its slice products in the         */
/* original order, so the result does not depend on the number of threads.       */
/* BNC_OZ_GEMM_MODE_BLAS keeps the old shape (one full DGEMM per slice pair,     */
/* left to the BLAS to thread).                                                  */
/*------------------------------------------------------------------------------*/
void mul_qdmatrix_oz(QDMatrix ret, QDMatrix a, int max_num_div_a, QDMatrix b, int max_num_div_b)
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
        fprintf(stderr, "ERROR: mul_qdmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
        return;
    }

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_b = (DMatrix *)calloc(max_num_div_b, sizeof(DMatrix));
    row_shift = (long int *)calloc((size_t)max_num_div_a * (size_t)row_dim, sizeof(long int));
    col_shift = (long int *)calloc((size_t)max_num_div_b * (size_t)col_dim, sizeof(long int));
    if(div_a == NULL || div_b == NULL || row_shift == NULL || col_shift == NULL)
    {
        fprintf(stderr, "ERROR: mul_qdmatrix_oz: cannot allocate\n");
        free(div_a); free(div_b); free(row_shift); free(col_shift);
        return;
    }
    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, mid_dim);
    for(i = 0; i < max_num_div_b; i++)
        div_b[i] = init_dmatrix(mid_dim, col_dim);

    real_num_div_a = split_qdmatrix_dmat_ex(div_a, row_shift, max_num_div_a, a);
    real_num_div_b = split_qdmatrix_t_dmat_ex(div_b, col_shift, max_num_div_b, b);

    set0_qdmatrix(ret);

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
            double *ret_row[QDSIZE];
            double ret_ij[QDSIZE], add_ij[QDSIZE], sum_ij[QDSIZE];

            for(comp = 0; comp < QDSIZE; comp++)
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

                            for(comp = 0; comp < QDSIZE; comp++)
                                ret_row[comp] = ret->element[comp] + (row_start + ii) * ret->real_col_dim;

                            for(jj = 0; jj < col_dim; jj++)
                            {
                                for(comp = 0; comp < QDSIZE; comp++)
                                    ret_ij[comp] = ret_row[comp][jj];
                                add_ij[0] = bnc_oz_ldexp(buf_row[jj], shift_a + shift_b[jj]);

                                rqd_add(sum_ij, ret_ij, add_ij);

                                for(comp = 0; comp < QDSIZE; comp++)
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
        double *ret_row[QDSIZE];
        double ret_ij[QDSIZE], add_ij[QDSIZE], sum_ij[QDSIZE];

        for(comp = 0; comp < QDSIZE; comp++)
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

                    for(comp = 0; comp < QDSIZE; comp++)
                        ret_row[comp] = ret->element[comp] + ii * ret->real_col_dim;

                    for(jj = 0; jj < col_dim; jj++)
                    {
                        for(comp = 0; comp < QDSIZE; comp++)
                            ret_ij[comp] = ret_row[comp][jj];
                        add_ij[0] = bnc_oz_ldexp(buf_row[jj], shift_a + shift_b[jj]);

                        rqd_add(sum_ij, ret_ij, add_ij);

                        for(comp = 0; comp < QDSIZE; comp++)
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
/* Same blocking as mul_qdmatrix_oz: a thread owns a range of rows of ret and */
/* runs all slice pairs for it, so the accumulation is parallel too.             */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
/* Matrix-Vector multiplication based on Ozaki scheme                            */
/*                                                                               */
/* Same blocking and the same exponent handling as mul_qdmatrix_oz(): a       */
/* thread owns a range of rows of ret and runs all slice pairs for it.           */
/*------------------------------------------------------------------------------*/
void mul_qdmatrix_qdvec_oz(QDVector ret, QDMatrix a, int max_num_div_a, QDVector vb, int max_num_div_vb)
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
        fprintf(stderr, "ERROR: mul_qdmatrix_qdvec_oz: cannot allocate\n");
        free(div_a); free(div_vb); free(row_shift); free(vec_shift);
        return;
    }

    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, col_dim);
    for(i = 0; i < max_num_div_vb; i++)
        div_vb[i] = init_dvector(vb->dim); // vb->dim, not ret->dim: they differ when a is not square

    real_num_div_a = split_qdmatrix_dmat_ex(div_a, row_shift, max_num_div_a, a);
    real_num_div_vb = split_qdvector_dvec_ex(div_vb, vec_shift, max_num_div_vb, vb);

    set0_qdvector(ret);

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
            double ret_i[QDSIZE], add_i[QDSIZE], sum_i[QDSIZE];

            for(comp = 0; comp < QDSIZE; comp++)
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

                            for(comp = 0; comp < QDSIZE; comp++)
                                ret_i[comp] = ret->element[comp][row_start + ii];
                            add_i[0] = bnc_oz_ldexp(buf[ii], shift_a + vec_shift[div_j]);

                            rqd_add(sum_i, ret_i, add_i);

                            for(comp = 0; comp < QDSIZE; comp++)
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
        double ret_i[QDSIZE], add_i[QDSIZE], sum_i[QDSIZE];

        for(comp = 0; comp < QDSIZE; comp++)
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

                    for(comp = 0; comp < QDSIZE; comp++)
                        ret_i[comp] = ret->element[comp][ii];
                    add_i[0] = bnc_oz_ldexp(div_ret->element[ii], shift_a + vec_shift[j]);

                    rqd_add(sum_i, ret_i, add_i);

                    for(comp = 0; comp < QDSIZE; comp++)
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
void mul_cqdmatrix_oz_3m(CQDMatrix ret, CQDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    QDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_qdmatrix(b->re->row_dim, ret->re->col_dim);

    mul_qdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    mul_qdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_b_image);
    sub_qdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_qdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_qdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        add_qdmatrix(ret->im, t1, t2);
    #else // USE_4M
    */
        // 3M
        add_qdmatrix(t3, a->re, a->im);
        add_qdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_qdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void mul_cqdmatrix_oz_4m(CQDMatrix ret, CQDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    QDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_qdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    mul_qdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_b_image);
    sub_qdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_qdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_qdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_b_image);
        add_qdmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_qdmatrix(t3, a->re, a->im);
        add_qdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_qdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */  

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}


// Testing
#ifdef DEBUG

//#define DIM 3
//#define DIM 10
//#define DIM 50
#define DIM 256

//#define MAX_NUM_DIV 2
#define MAX_NUM_DIV 10
//#define MAX_NUM_DIV 20

int main()
{
    long int i, j;
    int real_num_div;
    double tmp[QDSIZE], tmp2[QDSIZE];
    QDVector vec_org, vec, vec_c, vec_b;
    DVector dvec[MAX_NUM_DIV];
    QDMatrix mat_a, mat_b, mat_c, mat;
    DMatrix dmat[MAX_NUM_DIV];
    qdfloat ddtmp, ddtmp2;

    //goto qdmatrix;

// DVector 
    vec_org = init_qdvector(DIM);
    vec = init_qdvector(DIM);
    vec_b = init_qdvector(DIM);
    vec_c = init_qdvector(DIM);
    for(i = 0; i < MAX_NUM_DIV; i++)
        dvec[i] = init_dvector(DIM);

    // set random
    srand(DIM);
    for(i = 0; i < DIM; i++)
    {
        rqd_set_d(tmp, (double)rand());
        rqd_mul_d(tmp, tmp, (double)rand());
        rqd_div_d(tmp, tmp, (double)rand());
        set_qdvector_i(vec_org, i, tmp);
    }
    subst_qdvector(vec_b, vec_org);

    // split
    real_num_div = split_qdvector_dvec(dvec, MAX_NUM_DIV, vec_org); //, 53);

    for(i = 0; i < real_num_div; i++)
    {
        //printf("dvec[%d]:\n", i); print_dvector(dvec[i]);

        add_qdvector_dvec(vec, vec, dvec[i]);
    }
    //sub_qdvector(vec, vec, vec_org);
    //printf("vec - vec_org:\n"); print_qdvector(vec);
    //printf("\n");
    for(i = 0; i < DIM; i++)
    {
        ddtmp = get_qdvector_i_qdfloat(vec, i);
        ddtmp2 = get_qdvector_i_qdfloat(vec_org, i);
        printf("%5d %25.17e %25.17e %25.17e %25.17e\n%5d %25.17e %25.17e %25.17e %25.17e\n", i, ddtmp.val[0], ddtmp.val[1], ddtmp.val[2], ddtmp.val[3], i, ddtmp2.val[0], ddtmp2.val[1], ddtmp2.val[2], ddtmp2.val[3]);
    }
    
    //sgoto end;
// DMatrix
qdmatrix:

    mat_a = init_qdmatrix(DIM, DIM);
    mat_b = init_qdmatrix(DIM, DIM);
    mat_c = init_qdmatrix(DIM, DIM);
    mat = init_qdmatrix(DIM, DIM);
    for(i = 0; i < MAX_NUM_DIV; i++)
        dmat[i] = init_dmatrix(DIM, DIM);

    // set random
    srand(DIM);
    for(i = 0; i < DIM; i++)
    {
        for(j = 0; j < DIM; j++)
        {
            rqd_set_d(tmp, (double)rand() / (double)RAND_MAX);
            rqd_mul_d(tmp, tmp, (double)rand());
            rqd_div_d(tmp, tmp, (double)rand());
            set_qdmatrix_ij(mat_a, i, j, tmp);

            rqd_set_d(tmp, (double)rand() / (double)RAND_MAX);
            rqd_mul_d(tmp, tmp, (double)rand());
            rqd_div_d(tmp, tmp, (double)rand());
            set_qdmatrix_ij(mat_b, i, j, tmp);
        }
    }

    // split_A
    printf("Split_A:\n");
    real_num_div = split_qdmatrix_dmat(dmat, MAX_NUM_DIV, mat_a);
    printf("num_div, real_num_div = %d, %d\n", MAX_NUM_DIV, real_num_div);

    for(i = 0; i < real_num_div; i++)
    {
        //printf("dmat[%d]:\n", i); print_dmatrix(dmat[i]);

        add_qdmatrix_dmat(mat, mat, dmat[i]);
    }
    sub_qdmatrix(mat, mat, mat_a);
    printf("||mat - mat_org|| / ||mat_org||:\n"); 
    normf_qdmatrix(tmp, mat);
    normf_qdmatrix(tmp2, mat_a);
    rqd_div(tmp, tmp, tmp2);
    rqd_out_str(tmp); //print_qdmatrix(mat);
    printf("\n"); 

    // split_B
    printf("Split_B:\n");
    real_num_div = split_qdmatrix_t_dmat(dmat, MAX_NUM_DIV, mat_b);
    printf("num_div, real_num_div = %d, %d\n", MAX_NUM_DIV, real_num_div);

    for(i = 0; i < real_num_div; i++)
    {
        //printf("dmat[%d]:\n", i); print_dmatrix(dmat[i]);

        add_qdmatrix_dmat(mat, mat, dmat[i]);
    }
    sub_qdmatrix(mat, mat, mat_b);
    printf("||mat - mat_org|| / ||mat_org||:\n"); 
    normf_qdmatrix(tmp, mat);
    normf_qdmatrix(tmp2, mat_a);
    rqd_div(tmp, tmp, tmp2);
    rqd_out_str(tmp); //print_qdmatrix(mat);
    printf("\n"); 

    // a * b
    mul_qdmatrix_oz(mat, mat_a, MAX_NUM_DIV, mat_b, MAX_NUM_DIV);
    //mul_qdmatrix(mat_c, mat_a, mat_b);
    mul_qdmatrix_strassen(mat_c, mat_a, mat_b, 32);
/*
    for(i = 0; i < mat->row_dim; i++)
    {
        for(j = 0; j < mat->col_dim; j++)
        {
            ddtmp = get_qdmatrix_ij_qdfloat(mat, i, j);
            ddtmp2 = get_qdmatrix_ij_qdfloat(mat_c, i, j);
            if(ddtmp.val[1] != ddtmp2.val[1])
                printf("%5d,%5d %25.17e %25.17e %25.17e %25.17e\n%5d,%5d %25.17e %25.17e %25.17e %25.17e\n", i, j, ddtmp.val[0], ddtmp.val[1], ddtmp.val[2], ddtmp.val[3], i, j, ddtmp2.val[0], ddtmp2.val[1], ddtmp2.val[2], ddtmp2.val[3]);
        }
    }
*/
    sub_qdmatrix(mat, mat, mat_c);
    //print_qdmatrix(mat);
    printf("div = %ld\n", MAX_NUM_DIV);
    printf("||mat_oz - mat_org|| / ||mat_org||:\n"); 
    normf_qdmatrix(tmp, mat);
    normf_qdmatrix(tmp2, mat_c);
    rqd_div(tmp, tmp, tmp2);
    rqd_out_str(tmp); //print_qdmatrix(mat); 
    printf("\n"); 


    // a * vec
    mul_qdmatrix_qdvec_oz(vec, mat_a, MAX_NUM_DIV, vec_b, MAX_NUM_DIV);
    mul_qdmatrix_qdvec(vec_c, mat_a, vec_b);

    sub_qdvector(vec, vec, vec_c);
    //print_qdmatrix(mat);
    printf("div = %ld\n", MAX_NUM_DIV);
    printf("||vec_oz - vec_org|| / ||vec_org||:\n"); 
    norm2_qdvector(tmp, vec);
    norm2_qdvector(tmp2, vec_c);
    rqd_div(tmp, tmp, tmp2);
    rqd_out_str(tmp); //print_qdmatrix(mat); 
    printf("\n"); 

    free_qdmatrix(mat_a);
    free_qdmatrix(mat_b);
    free_qdmatrix(mat_c);
    free_qdmatrix(mat);
    for(i = 0; i < 10; i++)
        free_dmatrix(dmat[i]);

    free_qdvector(vec_org);
    free_qdvector(vec);
    for(i = 0; i < 10; i++)
        free_dvector(dvec[i]);

end:
    return 0;
}

// Testing
#endif // DEBUG
