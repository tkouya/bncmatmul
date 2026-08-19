/********************************************************************************/
/* oz_scheme: Multiple precision linear computation based on Ozaki scheme.      */
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

#include <stdlib.h>
#include <string.h>

// log2(x) := log10(x) / log10(2)
//#define DLOG2(x) (log10((x)) / 0.30102999566398119521373889472449)


/*------------------------------------------------------------------------------*/
/* Run-time configuration of the parallel Ozaki-scheme kernels.                  */
/*                                                                               */
/* The values are read from the environment once, on first use, and can be       */
/* overridden from the program with the bnc_oz_set_*() setters.  Reading them    */
/* lazily keeps the library free of constructors; the first call always happens  */
/* outside a parallel region (the kernels query the settings before they open    */
/* one), so no locking is needed.                                                */
/*------------------------------------------------------------------------------*/
static int _bnc_oz_config_done = 0;
static int _bnc_oz_num_threads = 0;        // 0 -> OpenMP maximum
static int _bnc_oz_blas_threads = 1;       // threads left to the BLAS backend
static long int _bnc_oz_block_rows = 0;    // 0 -> automatic
static int _bnc_oz_blocks_per_thread = 2;  // used when block_rows is automatic
static int _bnc_oz_gemm_mode = BNC_OZ_GEMM_MODE_OWN;

// minimum number of C rows worth handing to one DGEMM call
#define _BNC_OZ_MIN_BLOCK_ROWS 8

static long int _bnc_oz_getenv_long(const char *name, long int def_value)
{
    const char *str = getenv(name);
    char *end = NULL;
    long int value;

    if(str == NULL || *str == '\0')
        return def_value;

    value = strtol(str, &end, 10);
    if(end == str)
        return def_value;

    return value;
}

static void _bnc_oz_config(void)
{
    const char *mode;

    if(_bnc_oz_config_done)
        return;

    _bnc_oz_num_threads       = (int)_bnc_oz_getenv_long("BNC_OZ_NUM_THREADS", 0);
    _bnc_oz_blas_threads      = (int)_bnc_oz_getenv_long("BNC_OZ_BLAS_THREADS", 1);
    _bnc_oz_block_rows        = _bnc_oz_getenv_long("BNC_OZ_BLOCK_ROWS", 0);
    _bnc_oz_blocks_per_thread = (int)_bnc_oz_getenv_long("BNC_OZ_BLOCKS_PER_THREAD", 2);

    if(_bnc_oz_blocks_per_thread < 1)
        _bnc_oz_blocks_per_thread = 1;
    if(_bnc_oz_blas_threads < 1)
        _bnc_oz_blas_threads = 1;

    mode = getenv("BNC_OZ_GEMM_MODE");
    if(mode != NULL)
    {
        if(strcmp(mode, "blas") == 0 || strcmp(mode, "BLAS") == 0)
            _bnc_oz_gemm_mode = BNC_OZ_GEMM_MODE_BLAS;
        else if(strcmp(mode, "own") == 0 || strcmp(mode, "OWN") == 0)
            _bnc_oz_gemm_mode = BNC_OZ_GEMM_MODE_OWN;
    }

    _bnc_oz_config_done = 1;
}

// threads used by the Ozaki-scheme kernels
int bnc_oz_get_num_threads(void)
{
    _bnc_oz_config();

#ifdef _OPENMP
    if(_bnc_oz_num_threads > 0)
        return _bnc_oz_num_threads;

    return omp_get_max_threads();
#else // _OPENMP
    return 1;
#endif // _OPENMP
}

void bnc_oz_set_num_threads(int num_threads)
{
    _bnc_oz_config();
    _bnc_oz_num_threads = (num_threads > 0) ? num_threads : 0;
}

// threads left to the CBLAS backend inside a parallel kernel
int bnc_oz_get_blas_threads(void)
{
    _bnc_oz_config();

    return _bnc_oz_blas_threads;
}

void bnc_oz_set_blas_threads(int num_threads)
{
    _bnc_oz_config();
    _bnc_oz_blas_threads = (num_threads > 0) ? num_threads : 1;
}

// rows of C per block; <= 0 means "decide automatically"
long int bnc_oz_get_block_rows(void)
{
    _bnc_oz_config();

    return _bnc_oz_block_rows;
}

void bnc_oz_set_block_rows(long int block_rows)
{
    _bnc_oz_config();
    _bnc_oz_block_rows = (block_rows > 0) ? block_rows : 0;
}

// BNC_OZ_GEMM_MODE_OWN or BNC_OZ_GEMM_MODE_BLAS
int bnc_oz_get_gemm_mode(void)
{
    _bnc_oz_config();

#ifdef _OPENMP
    return _bnc_oz_gemm_mode;
#else // _OPENMP
    return BNC_OZ_GEMM_MODE_BLAS; // nothing to parallelize without OpenMP
#endif // _OPENMP
}

void bnc_oz_set_gemm_mode(int mode)
{
    _bnc_oz_config();
    _bnc_oz_gemm_mode = (mode == BNC_OZ_GEMM_MODE_BLAS) ? BNC_OZ_GEMM_MODE_BLAS : BNC_OZ_GEMM_MODE_OWN;
}

/*------------------------------------------------------------------------------*/
/* Rows of C handed to one block of the accumulation loop.                       */
/*                                                                               */
/* Several blocks per thread (four by default) let a dynamic schedule even out   */
/* cores of unequal speed, which is what a static split cannot do; too many      */
/* blocks, on the other hand, make every block re-read the whole B slice, so     */
/* the block is never smaller than _BNC_OZ_MIN_BLOCK_ROWS rows.                  */
/*------------------------------------------------------------------------------*/
long int bnc_oz_block_rows_for(long int row_dim, int num_threads)
{
    long int block_rows, num_blocks;

    _bnc_oz_config();

    if(_bnc_oz_block_rows > 0)
        block_rows = _bnc_oz_block_rows;
    else
    {
        if(num_threads < 1)
            num_threads = 1;

        num_blocks = (long int)num_threads * (long int)_bnc_oz_blocks_per_thread;
        block_rows = (row_dim + num_blocks - 1) / num_blocks;

        if(block_rows < _BNC_OZ_MIN_BLOCK_ROWS)
            block_rows = _BNC_OZ_MIN_BLOCK_ROWS;

        // keep the blocks aligned with the padding unit of the matrices
        block_rows = ((block_rows + _BNC_D_WIDTH - 1) / _BNC_D_WIDTH) * _BNC_D_WIDTH;
    }

    if(block_rows > row_dim)
        block_rows = row_dim;
    if(block_rows < 1)
        block_rows = 1;

    return block_rows;
}

/*------------------------------------------------------------------------------*/
/* BLAS thread control.                                                          */
/*                                                                               */
/* OpenBLAS (pthread build) and MKL both keep a global thread pool.  Calling     */
/* them from inside our own parallel region without saying anything would give   */
/* num_threads^2 workers and thrash; bnc_oz_blas_enter() therefore cuts the      */
/* backend down to BNC_OZ_BLAS_THREADS (1 by default) for the duration of the    */
/* kernel and bnc_oz_blas_leave() restores whatever the caller had set.  Both    */
/* are no-ops when the library was built without a CBLAS backend.                */
/*------------------------------------------------------------------------------*/
int bnc_oz_blas_enter(void)
{
#if defined(USE_IMKL)
    int prev = mkl_get_max_threads();

    mkl_set_num_threads(bnc_oz_get_blas_threads());

    return prev;
#elif defined(USE_OPENBLAS) // USE_IMKL
    int prev = openblas_get_num_threads();

    openblas_set_num_threads(bnc_oz_get_blas_threads());

    return prev;
#else // USE_IMKL
    return 0;
#endif // USE_IMKL
}

void bnc_oz_blas_leave(int prev_num_threads)
{
#if defined(USE_IMKL)
    if(prev_num_threads > 0)
        mkl_set_num_threads(prev_num_threads);
#elif defined(USE_OPENBLAS) // USE_IMKL
    if(prev_num_threads > 0)
        openblas_set_num_threads(prev_num_threads);
#else // USE_IMKL
    (void)prev_num_threads;
#endif // USE_IMKL
}

// exponent e of x, so that |x| is in [2^(e-1), 2^e); 0 for zero and non-finite x
long int bnc_oz_exp2_d(double x)
{
    int exponent = 0;

    if(x == 0.0 || isnan(x) || isinf(x))
        return 0;

    frexp(x, &exponent); // |x| = mantissa * 2^exponent, mantissa in [0.5, 1)

    return (long int)exponent;
}

// ldexp() with the shift saturated (see oz_scheme.h)
double bnc_oz_ldexp(double x, long int shift)
{
    if(shift > 2200)
        shift = 2200;      // overflows to +-Inf either way
    else if(shift < -2200)
        shift = -2200;     // underflows to 0 either way

    return ldexp(x, (int)shift);
}

/*------------------------------------------------------------------------------*/
/* ret_block := a[row_start : row_start + num_rows][*] * b                       */
/*                                                                               */
/* Only the logical dimensions take part: the padding columns of a and the       */
/* padding rows of b are zero, so leaving them out changes nothing but saves     */
/* the work.  The products of two split matrices are error free, hence the       */
/* summation order is irrelevant and the fallback loop below returns exactly     */
/* what the CBLAS call returns.                                                  */
/*------------------------------------------------------------------------------*/
void bnc_oz_dgemm_block(double *ret_block, long int ld_ret_block, DMatrix a, long int row_start, long int num_rows, DMatrix b)
{
    long int i, j, mid_dim = a->col_dim, col_dim = b->col_dim;

    if(num_rows <= 0)
        return;

#ifdef BNC_USE_CBLAS
    for(i = 0; i < num_rows; i++)
        for(j = 0; j < col_dim; j++)
            ret_block[i * ld_ret_block + j] = 0.0;

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        num_rows,               // m
        col_dim,                // n
        mid_dim,                // k
        1.0,
        a->element + row_start * a->real_col_dim,
        a->real_col_dim,        // lda
        b->element,
        b->real_col_dim,        // ldb
        1.0,
        ret_block,
        ld_ret_block            // ldc
    );
#else // BNC_USE_CBLAS
    long int k;

    for(i = 0; i < num_rows; i++)
    {
        const double *a_row = a->element + (row_start + i) * a->real_col_dim;
        double *ret_row = ret_block + i * ld_ret_block;

        for(j = 0; j < col_dim; j++)
            ret_row[j] = 0.0;

        for(k = 0; k < mid_dim; k++)
        {
            const double *b_row = b->element + k * b->real_col_dim;
            double a_ik = a_row[k];

            if(a_ik == 0.0)
                continue;

            for(j = 0; j < col_dim; j++)
                ret_row[j] += a_ik * b_row[j];
        }
    }
#endif // BNC_USE_CBLAS
}

// ret_block := a[row_start : row_start + num_rows][*] * b
void bnc_oz_dgemv_block(double *ret_block, DMatrix a, long int row_start, long int num_rows, DVector b)
{
    long int i, mid_dim = a->col_dim;

    if(num_rows <= 0)
        return;

#ifdef BNC_USE_CBLAS
    for(i = 0; i < num_rows; i++)
        ret_block[i] = 0.0;

    cblas_dgemv(
        CblasRowMajor,
        CblasNoTrans,
        num_rows,               // m
        mid_dim,                // n
        1.0,
        a->element + row_start * a->real_col_dim,
        a->real_col_dim,        // lda
        b->element,
        1,
        1.0,
        ret_block,
        1
    );
#else // BNC_USE_CBLAS
    long int j;

    for(i = 0; i < num_rows; i++)
    {
        const double *a_row = a->element + (row_start + i) * a->real_col_dim;
        double sum = 0.0;

        for(j = 0; j < mid_dim; j++)
            sum += a_row[j] * b->element[j];

        ret_block[i] = sum;
    }
#endif // BNC_USE_CBLAS
}

// extract vector
// ret_high + ret_low = org_vec
void extract_dvector(DVector ret_high_vec, DVector ret_low_vec, DVector org_vec, double num_bits)
{
    //long int dim = org_vec->dim;
    long int dim = org_vec->real_dim;
    long int i;
    double org_vec_i, ret_high_vec_i, absmax_org_vec, tmp;
    double threshold, t_exp;

	absmax_org_vec = absmax_dvector(NULL, org_vec);

    // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
	t_exp = ceil(DLOG2(absmax_org_vec)) + ceil(((double)num_bits + DLOG2((double)(dim + 1))) / 2.0);
    threshold = pow(2.0, t_exp);

    for(i = 0; i < dim; i++)
    {
        // set high vector
        org_vec_i = get_dvector_i(org_vec, i);
        ret_high_vec_i = org_vec_i + threshold;
        ret_high_vec_i -= threshold;
        set_dvector_i(ret_high_vec, i, ret_high_vec_i);

        // set low vector
        set_dvector_i(ret_low_vec, i, org_vec_i - ret_high_vec_i);
    }
}

// SplitMat_A
void split_dmatrix(DMatrix ret_high_mat, DMatrix ret_low_mat, DMatrix org_mat)
{
	long int i, j, row_dim, col_dim, real_total_dim;
	long int num_digits = 53; // IEEE double prec.
	//double *s;
    DMatrix s;
	double mu, abs_aij, t_exp;

    //row_dim = org_mat->row_dim;
    //col_dim = org_mat->col_dim;
    row_dim = org_mat->real_row_dim;
    col_dim = org_mat->real_col_dim;
    real_total_dim = row_dim * col_dim;

	// threshold matrix 
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    s = init_dmatrix(row_dim, col_dim);

	// mu[i] = max_j |mat[i, j]|
	for(i = 0; i < row_dim; i++)
	{
		//mu = fabs(mat[i * col_dim + 0]);
		/*for(j = 1; j < col_dim; j++)
		{
			abs_aij = fabs(mat[i * col_dim + j]);
			if(abs_aij > mu)
				mu = abs_aij;
		}*/
        mu = absmax_row_dmatrix(NULL, i, org_mat);

		// t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
		t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);

		// s[i, j] = 2^t_exp
		for(j = 0; j < col_dim; j++)
        {
			//s[i * col_dim + j] = pow(2.0, t_exp);
            set_dmatrix_ij(s, i, j, pow(2.0, t_exp));
        }
	}

// split org_mat to ret_high_mat and ret_low_mat
#ifdef USE_CBLAS
	// tmp_mat := mat + s
	cblas_dcopy(real_total_dim, org_mat->element, 1, ret_high_mat->element, 1);
	cblas_daxpy(real_total_dim, 1.0, s->element, 1, ret_high_mat->element, 1);

	// high_mat := tmp_mat - s
	cblas_daxpy(real_total_dim, -1.0, s->element, 1, ret_high_mat->element, 1);

	// low_mat := mat - high_mat
	cblas_dcopy(real_total_dim, mat->element, 1, ret_low_mat->element, 1);
	cblas_daxpy(real_total_dim, -1.0, ret_high_mat->element, 1, ret_low_mat->element, 1);
#else // USE_CBLAS
	// tmp_mat := mat + s
	add_dmatrix(ret_high_mat, org_mat, s);

	// high_mat := tmp_mat - s
	sub_dmatrix(ret_high_mat, ret_high_mat, s);

	// low_mat := mat - high_mat
    sub_dmatrix(ret_low_mat, org_mat, ret_high_mat);
#endif // USE_CBLAS

	// free s
	free_dmatrix(s);
}

// SplitMat_B
void split_dmatrix_t(DMatrix ret_high_mat, DMatrix ret_low_mat, DMatrix org_mat)
{
	long int i, j, row_dim, col_dim, real_total_dim;
	int num_digits = 53; // IEEE double prec.
	//double *s;
    DMatrix s;
	double mu, abs_aij, t_exp;

    //row_dim = org_mat->row_dim;
    //col_dim = org_mat->col_dim;
    row_dim = org_mat->real_row_dim;
    col_dim = org_mat->real_col_dim;
    real_total_dim = row_dim * col_dim;

	// initialize s
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    s = init_dmatrix(row_dim, col_dim);

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
        mu = absmax_col_dmatrix(NULL, j, org_mat);

		// t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
		t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);

		// s[i, j] = 2^t_exp
		for(i = 0; i < row_dim; i++)
            set_dmatrix_ij(s, i, j, pow(2.0, t_exp));
			//s[i * col_dim + j] = pow(2.0, t_exp);
	}

#ifdef USE_CBLAS
	// tmp_mat := mat + s
	cblas_dcopy(real_total_dim, org_mat->element, 1, ret_high_mat->element, 1);
	cblas_daxpy(real_total_dim, 1.0, s, 1, ret_high_mat->element 1);

	// high_mat := tmp_mat - s
	cblas_daxpy(real_total_dim, -1.0, s->element, 1, ret_high_mat->element, 1);

	// low_mat := mat - high_mat
	cblas_dcopy(real_total_dim, org_mat->element, 1, ret_low_mat->element, 1);
	cblas_daxpy(real_total_dim, -1.0, ret_high_mat->element, 1, ret_low_mat->element, 1);
#else // USE_CBLAS
	// tmp_mat := mat + s
    add_dmatrix(ret_high_mat, org_mat, s);

	// high_mat := tmp_mat - s
    sub_dmatrix(ret_high_mat, ret_high_mat, s);

	// low_mat := mat - high_mat
    sub_dmatrix(ret_low_mat, org_mat, ret_high_mat);

#endif // USE_CBLAS

    // free s
	free_dmatrix(s);
}


// split vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
//void extract_dvector(DVector ret_vec[], int num_div, DVector org_vec, int num_bits)
/*------------------------------------------------------------------------------*/
/* split vector: 2^shift[index] * (ret_vec[0] + ... ) = org_vec                  */
/*                                                                               */
/* The vector plays the role of a single column of B, so it carries one exponent */
/* per slice.  shift may be NULL; see oz_scheme.h.                               */
/*------------------------------------------------------------------------------*/
int split_dvector_dvec_ex(DVector ret_vec[], long int shift[], int num_div, DVector org_vec)
{
    long int dim = org_vec->dim, i;
    int index, num_bits = 53, real_num_div; // IEEE754 binary64
    double org_vec_i, high_i, absmax_org_vec, threshold, tail_exp;
    long int sigma;
    DVector tmp_org_vec, own_ret_vec = NULL, in_ret_vec;

    if(ret_vec == NULL)
    {
        own_ret_vec = init_dvector(dim);
        in_ret_vec = own_ret_vec;
    }
    else
        in_ret_vec = ret_vec[0];

    tmp_org_vec = init_dvector(dim);
    subst_dvector(tmp_org_vec, org_vec);

    tail_exp = ceil(((double)num_bits + DLOG2((double)(dim))) / 2.0);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        if(ret_vec != NULL)
            in_ret_vec = ret_vec[index];

        absmax_org_vec = absmax_dvector(NULL, tmp_org_vec);

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

        threshold = pow(2.0, ceil(DLOG2(absmax_org_vec)) + tail_exp);

        for(i = 0; i < dim; i++)
        {
            org_vec_i = get_dvector_i(tmp_org_vec, i);
            if(sigma != 0)
                org_vec_i = bnc_oz_ldexp(org_vec_i, -sigma);

            high_i = org_vec_i + threshold;
            high_i = high_i - threshold;
            set_dvector_i(in_ret_vec, i, high_i);

            // low vector := vector - 2^shift * high
            set_dvector_i(tmp_org_vec, i,
                get_dvector_i(tmp_org_vec, i) - ((sigma != 0) ? bnc_oz_ldexp(high_i, sigma) : high_i));
        }

        real_num_div = index + 1;
    }

    free_dvector(tmp_org_vec);
    if(own_ret_vec != NULL)
        free_dvector(own_ret_vec);

    return real_num_div;
}

// split vector without the scaling
int split_dvector_dvec(DVector ret_vec[], int num_div, DVector org_vec)
{
    return split_dvector_dvec_ex(ret_vec, NULL, num_div, org_vec);
}

// SplitMat_A
// return real_num_div
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
/* SplitMat_A: 2^row_shift[index][i] * (ret_mat[0] + ret_mat[1] + ...) = org_mat,*/
/* every ret_mat[] a plain double matrix whose rows multiply without rounding.   */
/*                                                                               */
/* Two passes over the data per split instead of the six the straightforward     */
/* formulation needs, and both run row-parallel because rows never interact.     */
/* The threshold is constant along a row, so a vector of row thresholds replaces */
/* the full s matrix that used to be built and streamed for every split.         */
/*                                                                               */
/* row_shift may be NULL, which asks for the unscaled split; see oz_scheme.h.    */
/*------------------------------------------------------------------------------*/
int split_dmatrix_dmat_ex(DMatrix ret_mat[], long int row_shift[], int num_div, DMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, org_stride, ret_stride;
	long int num_digits = 53; // IEEE double prec.
	int real_num_div, num_threads;
	double mu_total, tail_exp;
	DMatrix tmp_org_mat, own_ret_mat = NULL, in_ret_mat;
	double *power2; // 2^t_exp of each row, in the scaled domain
	long int *shift; // 2^shift of each row of this slice

	row_dim = org_mat->row_dim;
	col_dim = org_mat->col_dim;

	if(ret_mat != NULL)
	{
		if((ret_mat[0]->row_dim != row_dim) || (ret_mat[0]->col_dim != col_dim))
		{
			fprintf(stderr, "ERROR: split_dmatrix_dmat\n");
			return 0;
		}
	}

	power2 = (double *)calloc((size_t)row_dim, sizeof(double));
	if(power2 == NULL)
	{
		fprintf(stderr, "ERROR: split_dmatrix_dmat: cannot allocate\n");
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
	tmp_org_mat = init_dmatrix(row_dim, col_dim);
	subst_dmatrix(tmp_org_mat, org_mat);
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

		// pass 1: ret_mat[index] := 2^-shift[i] * tmp_org_mat and mu[i] = max_j |ret_mat[index][i][j]|
		mu_total = 0.0;

#ifdef _OPENMP
		#pragma omp parallel for num_threads(num_threads) schedule(static) private(i, j) reduction(+:mu_total)
#endif // _OPENMP
		for(i = 0; i < row_dim; i++)
		{
			const double *org_row = tmp_org_mat->element + i * org_stride;
			double *ret_row = in_ret_mat->element + i * ret_stride;
			double mu = 0.0, abs_org_ij;
			long int sigma = 0;

			for(j = 0; j < col_dim; j++)
			{
				abs_org_ij = fabs(org_row[j]);
				if(abs_org_ij > mu)
					mu = abs_org_ij;
			}

			// normalize the row into [1/2, 1) so that neither the conversion nor
			// the threshold below can leave the double exponent range
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
			double *org_row = tmp_org_mat->element + i * org_stride;
			double s = power2[i], high_ij;
			long int sigma = (shift != NULL) ? shift[i] : 0;

			for(j = 0; j < col_dim; j++)
			{
				// (x + s) - s keeps the leading bits of x; valid under the IEEE
				// semantics this library is compiled with (no -ffast-math)
				high_ij = ret_row[j] + s;
				high_ij = high_ij - s;
				ret_row[j] = high_ij;

				// low_mat := mat - high_mat
				org_row[j] = org_row[j] - ((sigma != 0) ? bnc_oz_ldexp(high_ij, sigma) : high_ij);
			}
		}

		real_num_div = index + 1;
	}

	free(power2);
	free_dmatrix(tmp_org_mat);
	if(own_ret_mat != NULL)
		free_dmatrix(own_ret_mat);

	return real_num_div;
}

// SplitMat_A without the scaling; kept for callers that cannot apply a scale factor
int split_dmatrix_dmat(DMatrix ret_mat[], int num_div, DMatrix org_mat)
{
	return split_dmatrix_dmat_ex(ret_mat, NULL, num_div, org_mat);
}

// SplitMat_B
// return real_num_div
/*------------------------------------------------------------------------------*/
/* SplitMat_B: same as split_dmatrix_dmat, but the threshold of the second       */
/* operand is taken over columns, so the maxima are reduced per column -- one    */
/* partial maximum vector per thread, combined afterwards, so that the sweep     */
/* can still walk the matrix row-wise.                                           */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
/* SplitMat_B: same as split_dmatrix_dmat_ex(), but the threshold and the scale  */
/* of the second operand are taken over columns, so the maxima are reduced per   */
/* column -- one partial maximum vector per thread, combined afterwards, so that */
/* the sweep can still walk the matrix row-wise.                                 */
/*------------------------------------------------------------------------------*/
int split_dmatrix_t_dmat_ex(DMatrix ret_mat[], long int col_shift[], int num_div, DMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, org_stride, ret_stride;
	int real_num_div, num_digits = 53, num_threads, thread; // IEEE double prec.
	double mu_total, tail_exp;
	DMatrix tmp_org_mat, own_ret_mat = NULL, in_ret_mat;
	double *power2, *mu_local; // 2^t_exp of each column, per-thread column maxima
	long int *shift;

	row_dim = org_mat->row_dim;
	col_dim = org_mat->col_dim;

	if(ret_mat != NULL)
	{
		if((ret_mat[0]->row_dim != row_dim) || (ret_mat[0]->col_dim != col_dim))
		{
			fprintf(stderr, "ERROR: split_dmatrix_t_dmat\n");
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
		fprintf(stderr, "ERROR: split_dmatrix_t_dmat: cannot allocate\n");
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

	tmp_org_mat = init_dmatrix(row_dim, col_dim);
	subst_dmatrix(tmp_org_mat, org_mat);
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
				const double *org_row = tmp_org_mat->element + i * org_stride;
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
			double *org_row = tmp_org_mat->element + i * org_stride;
			double s, high_ij, scaled_ij;
			long int sigma;

			for(j = 0; j < col_dim; j++)
			{
				sigma = (shift != NULL) ? shift[j] : 0;
				s = power2[j];

				scaled_ij = (sigma != 0) ? bnc_oz_ldexp(org_row[j], -sigma) : org_row[j];

				high_ij = scaled_ij + s;
				high_ij = high_ij - s;
				ret_row[j] = high_ij;

				org_row[j] = org_row[j] - ((sigma != 0) ? bnc_oz_ldexp(high_ij, sigma) : high_ij);
			}
		}

		real_num_div = index + 1;
	}

	free(power2);
	free(mu_local);
	free_dmatrix(tmp_org_mat);
	if(own_ret_mat != NULL)
		free_dmatrix(own_ret_mat);

	return real_num_div;
}

// SplitMat_B without the scaling; kept for callers that cannot apply a scale factor
int split_dmatrix_t_dmat(DMatrix ret_mat[], int num_div, DMatrix org_mat)
{
	return split_dmatrix_t_dmat_ex(ret_mat, NULL, num_div, org_mat);
}

/*------------------------------------------------------------------------------*/
/* Matrix multiplication based on Ozaki scheme                                   */
/*                                                                               */
/* The double-precision member of the mul_{dd,td,qd,mpf}matrix_oz() family, with */
/* the same blocking: the rows of ret are cut into blocks, one thread takes a    */
/* block at a time and runs every slice product for it with a single-threaded    */
/* DGEMM, accumulating on the spot.  Blocks are disjoint and each element of ret */
/* still sums its slice products in the original order, so the result does not   */
/* depend on the number of threads.                                              */
/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
/* Matrix multiplication based on Ozaki scheme                                   */
/*                                                                               */
/* The double-precision member of the mul_{dd,td,qd,mpf}matrix_oz() family, with */
/* the same blocking: the rows of ret are cut into blocks, one thread takes a    */
/* block at a time and runs every slice product for it with a single-threaded    */
/* DGEMM, accumulating on the spot.  Blocks are disjoint and each element of ret */
/* still sums its slice products in the original order, so the result does not   */
/* depend on the number of threads.                                              */
/*                                                                               */
/* Each slice carries the power of two its row (of A) or column (of B) was       */
/* scaled by, and the two exponents are put back on the product here; see the    */
/* exponent-handling note in oz_scheme.h.                                        */
/*------------------------------------------------------------------------------*/
void mul_dmatrix_oz(DMatrix ret, DMatrix a, int max_num_div_a, DMatrix b, int max_num_div_b)
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
        fprintf(stderr, "ERROR: mul_dmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
        return;
    }

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_b = (DMatrix *)calloc(max_num_div_b, sizeof(DMatrix));
    row_shift = (long int *)calloc((size_t)max_num_div_a * (size_t)row_dim, sizeof(long int));
    col_shift = (long int *)calloc((size_t)max_num_div_b * (size_t)col_dim, sizeof(long int));
    if(div_a == NULL || div_b == NULL || row_shift == NULL || col_shift == NULL)
    {
        fprintf(stderr, "ERROR: mul_dmatrix_oz: cannot allocate\n");
        free(div_a); free(div_b); free(row_shift); free(col_shift);
        return;
    }
    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, mid_dim);
    for(i = 0; i < max_num_div_b; i++)
        div_b[i] = init_dmatrix(mid_dim, col_dim);

    real_num_div_a = split_dmatrix_dmat_ex(div_a, row_shift, max_num_div_a, a);
    real_num_div_b = split_dmatrix_t_dmat_ex(div_b, col_shift, max_num_div_b, b);

    set0_dmatrix(ret);

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
                            double *ret_row = ret->element + (row_start + ii) * ret->real_col_dim;
                            const double *buf_row = buf + ii * col_dim;

                            shift_a = row_shift[(size_t)div_i * (size_t)row_dim + row_start + ii];

                            for(jj = 0; jj < col_dim; jj++)
                                ret_row[jj] += bnc_oz_ldexp(buf_row[jj], shift_a + shift_b[jj]);
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

        div_ret = init_dmatrix(row_dim, col_dim);

        for(i = 0; i < real_num_div_a; i++)
        {
            for(j = 0; j < real_num_div_b - i; j++)
            {
                const long int *shift_b = col_shift + (size_t)j * (size_t)col_dim;

                bnc_oz_dgemm_block(div_ret->element, div_ret->real_col_dim, div_a[i], 0, row_dim, div_b[j]);

                for(ii = 0; ii < row_dim; ii++)
                {
                    double *ret_row = ret->element + ii * ret->real_col_dim;
                    const double *buf_row = div_ret->element + ii * div_ret->real_col_dim;
                    long int shift_a = row_shift[(size_t)i * (size_t)row_dim + ii];

                    for(jj = 0; jj < col_dim; jj++)
                        ret_row[jj] += bnc_oz_ldexp(buf_row[jj], shift_a + shift_b[jj]);
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

// Testing
#ifdef DEBUG

#include "dd_oz_scheme.c"

#define DIM 5
//#define DIM 10

int main()
{
    long int i, j;
    DVector dvec_org, dvec[10];
    DMatrix dmat_org, dmat[10];

// DVector 

    dvec_org = init_dvector(DIM);
    for(i = 0; i < 10; i++)
        dvec[i] = init_dvector(DIM);

    // set random
    srand(DIM);
    for(i = 0; i < DIM; i++)
        set_dvector_i(dvec_org, i, (double)rand() * rand());

    // split
    extract_dvector(dvec[0], dvec[1], dvec_org, 53);

    printf("dvec[0]:\n"); print_dvector(dvec[0]);
    printf("dvec[1]:\n"); print_dvector(dvec[1]);
    printf("dvec_org:\n"); print_dvector(dvec_org);
    add_dvector(dvec[2], dvec[0], dvec[1]);
    sub_dvector(dvec[3], dvec[2], dvec_org);
    printf("dvec[0] + dvec[1] - dvec_org:^b"); print_dvector(dvec[3]); 

    free_dvector(dvec_org);
    for(i = 0; i < 10; i++)
        free_dvector(dvec[i]);

// DMatrix

    dmat_org = init_dmatrix(DIM, DIM);
    for(i = 0; i < 10; i++)
        dmat[i] = init_dmatrix(DIM, DIM);

    // set random
    srand(DIM);
    for(i = 0; i < DIM; i++)
        for(j = 0; j < DIM; j++)
            set_dmatrix_ij(dmat_org, i, j, (double)rand() * rand());

    // split
    split_dmatrix(dmat[0], dmat[1], dmat_org);

    printf("dmat[0]:\n"); print_dmatrix(dmat[0]);
    printf("dmat[1]:\n"); print_dmatrix(dmat[1]);
    printf("dmat_org:\n"); print_dmatrix(dmat_org);
    add_dmatrix(dmat[2], dmat[0], dmat[1]);
    sub_dmatrix(dmat[3], dmat[2], dmat_org);
    printf("dmat[0] + dmat[1] - dmat_org:\n"); print_dmatrix(dmat[3]); 

    free_dmatrix(dmat_org);
    for(i = 0; i < 10; i++)
        free_dmatrix(dmat[i]);


    return 0;
}

#endif // DEBUG
