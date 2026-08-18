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
int split_mpfvector_dvec(DVector ret_vec[], int num_div, MPFVector org_vec) //, int num_bits)
{
    unsigned int prec = org_vec->prec;
    long int dim = org_vec->dim;
    int index, real_num_div, num_bits = 53; // IEEE754 binary64
    long int i;
    double org_vec_i, ret_high_vec_i;
    mpf_t tmp;
    double absmax_org_vec, threshold, t_exp; 
    MPFVector tmp_org_vec;

    mpf_init2(tmp, prec);

    // tmp_org_vec := org_vec
    tmp_org_vec = init2_mpfvector(dim, prec);
    subst_mpfvector(tmp_org_vec, org_vec);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        subst_dvector_mpfvec(ret_vec[index], tmp_org_vec);
        absmax_org_vec = absmax_dvector(NULL, ret_vec[index]);

        // ret_vec[index] == 0
        if(absmax_org_vec == 0.0) break;
        // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
        //t_exp = ceil(DLOG2(absmax_org_vec)) + ceil(((double)num_bits + DLOG2((double)(dim + 1))) / 2.0);
        t_exp = ceil(DLOG2(absmax_org_vec)) + ceil(((double)num_bits + DLOG2((double)(dim))) / 2.0);

        //mpf_pow(threshold, two, t_exp);
        //mpf_pow_mpfr(threshold, two, t_exp);
        threshold = pow(2.0, t_exp);

        for(i = 0; i < dim; i++)
        {
            // set high vector
            //mpf_set(org_vec_i, get_mpfvector_i(tmp_org_vec, i)); 
            org_vec_i = get_dvector_i(ret_vec[index], i);   
            ret_high_vec_i = org_vec_i + threshold;
            //mpf_add(ret_high_vec_i, org_vec_i, threshold);
            ret_high_vec_i -= threshold;
            //mpf_sub(ret_high_vec_i, ret_high_vec_i, threshold);
            set_dvector_i(ret_vec[index], i, ret_high_vec_i);

            // set low vector
            mpf_sub_d(tmp, get_mpfvector_i(tmp_org_vec, i), ret_high_vec_i);           
            set_mpfvector_i(tmp_org_vec, i, tmp);
        }

        real_num_div = index + 1;
    }

    free_mpfvector(tmp_org_vec);
    mpf_clear(tmp);

    return real_num_div;
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
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;
	mpf_t tmp;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_mpfmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_mpfmatrix_dmat\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;
    mpf_init2(tmp, c->prec);

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

/* c := a - (doble)b */
void sub_mpfmatrix_dmat(MPFMatrix c, MPFMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;
    mpf_t tmp;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_mpfmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_mpfmatrix_dmat\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;
	mpf_init2(tmp, c->prec);

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

//#define SPLIT_NUM_DIGITS 64

// SplitMat_A
int split_mpfmatrix_dmat(DMatrix ret_mat[], int num_div, MPFMatrix org_mat)
{
    unsigned long prec = org_mat->prec;
	long int i, j, index, row_dim, col_dim, real_total_dim;
	long int num_digits = 53; // IEEE double prec.
	//long int num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//double *s;
    int real_num_div;
    MPFMatrix tmp_org_mat;
    DMatrix s, tmp_mat[2]; 
	double mu, abs_aij, t_exp, power2, mu_total;

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;

	// threshold matrix 
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    //s = init_mpfmatrix(row_dim, col_dim);
    s = init_dmatrix(row_dim, col_dim);

    //tmp_mat[0] = init_mpfmatrix(row_dim, col_dim);
    //tmp_mat[1] = init_mpfmatrix(row_dim, col_dim);
    tmp_mat[0] = init_dmatrix(row_dim, col_dim);
    tmp_mat[1] = init_dmatrix(row_dim, col_dim);

    // tmp_org_mat := org_mat;
    tmp_org_mat = init2_mpfmatrix(row_dim, col_dim, prec);
    subst_mpfmatrix(tmp_org_mat, org_mat);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        //printf("In split_mpfmatrix ... index= %d\n", index);
        //set0_dmatrix(ret_mat[index]);
        subst_dmatrix_mpfmat(ret_mat[index], tmp_org_mat);

        // mu[i] = max_j |mat[i, j]|
        // mu_total += mu
        mu_total = 0.0;
        for(i = 0; i < row_dim; i++)
        {
            //absmax_row_mpfmatrix(mu, NULL, i, tmp_org_mat);
            mu = absmax_row_dmatrix(NULL, i, ret_mat[index]);
            mu_total += mu;

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            //t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);
            //t_exp[1] = 0.0;
            t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(col_dim))) / 2.0);

            // s[i, j] = 2^t_exp
            //mpf_pow(power2, two, t_exp);
            //mpf_pow_mpfr(power2, two, t_exp);
            power2 = pow(2.0, t_exp);
            //printf("index, i, power2 = %ld, %ld, %25.17e\n", index, i, power2[0]);
            for(j = 0; j < col_dim; j++)
            {
                //s[i * col_dim + j] = pow(2.0, t_exp);
                //set_mpfmatrix_ij(s, i, j, power2);
                set_dmatrix_ij(s, i, j, power2);
            }
        }

        // ret_mat[index] == 0
        if(mu_total == 0.0) break;

        // split org_mat to ret_high_mat and ret_low_mat
#ifdef USE_IMKL
        real_total_dim = ret_mat[index]->real_row_dim * ret_mat[index]->real_col_dim;

        // tmp_mat := mat + s
        //blas_dcopy(real_total_dim, ret_mat[index]->element, 1, tmp_mat[0]->element, 1);
        //cblas_daxpy(real_total_dim, 1.0, s->element, 1, tmp_mat[0]->element, 1);
        cblas_daxpy(real_total_dim, 1.0, s->element, 1, ret_mat[index]->element, 1);

        // high_mat := tmp_mat - s
        cblas_daxpy(real_total_dim, -1.0, s->element, 1, ret_mat[index]->element, 1);

        // low_mat := mat - high_mat
        //cblas_dcopy(real_total_dim, tmp_mat[0]->element, 1, ret_mat[index]->element, 1);
#else // USE_IMKL
        // tmp_mat := mat + s
        //add_mpfmatrix(tmp_mat[0], tmp_org_mat, s);
        add_dmatrix(tmp_mat[0], ret_mat[index], s);

        // high_mat := tmp_mat - s
        //sub_mpfmatrix(tmp_mat[1], tmp_mat[0], s);
        sub_dmatrix(tmp_mat[1], tmp_mat[0], s);
        subst_dmatrix(ret_mat[index], tmp_mat[1]);
#endif // USE_IMKL

        // low_mat := mat - high_mat
        sub_mpfmatrix_dmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
        //sub_mpfmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);

        real_num_div = index + 1;
    }

	// free s
	//free_mpfmatrix(s);
	free_dmatrix(s);
    free_mpfmatrix(tmp_org_mat);
    free_dmatrix(tmp_mat[0]);
    free_dmatrix(tmp_mat[1]);

    return real_num_div;
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
int split_mpfmatrix_t_dmat(DMatrix ret_mat[], int num_div, MPFMatrix org_mat)
{
    unsigned long prec = org_mat->prec;

	long int i, j, index, row_dim, col_dim, real_total_dim;
	int real_num_div, num_digits = 53; // IEEE double prec.
	//int flag_stop = 0, num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//double *s;
    MPFMatrix tmp_org_mat;
    DMatrix s, tmp_mat[2];
	//double mu[TDSIZE], abs_aij[TDSIZE], t_exp[TDSIZE], power2[TDSIZE], two[TDSIZE] = {2.0, 0.0};
	double mu, abs_aij, t_exp, power2, mu_total;

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;

	// initialize s
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    //s = init_mpfmatrix(row_dim, col_dim);
    s = init_dmatrix(row_dim, col_dim);

    tmp_mat[0] = init_dmatrix(row_dim, col_dim);
    tmp_mat[1] = init_dmatrix(row_dim, col_dim);

    tmp_org_mat = init2_mpfmatrix(row_dim, col_dim, prec);
    subst_mpfmatrix(tmp_org_mat, org_mat);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        subst_dmatrix_mpfmat(ret_mat[index], tmp_org_mat);

        // mu[j] = max_j |mat[i, j]|
        // mu_total += mu
        mu_total = 0.0;
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
            mu = absmax_col_dmatrix(NULL, j, ret_mat[index]);
            mu_total += mu;

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            //t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);
            //t_exp[1] = 0.0;
            t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(row_dim))) / 2.0);

            // s[i, j] = 2^t_exp
            //mpf_pow(power2, two, t_exp);
            //mpf_pow_mpfr(power2, two, t_exp);
            power2 = pow(2.0, t_exp);

            //printf("index, j, power2 = %ld, %ld, %25.17e\n", index, j, power2[0]);
            for(i = 0; i < row_dim; i++)
                set_dmatrix_ij(s, i, j, power2);
                //s[i * col_dim + j] = pow(2.0, t_exp);
        }
        // ret_mat[index] == 0
        if(mu_total == 0.0) break;

#ifdef USE_IMKL
        real_total_dim = ret_mat[index]->real_row_dim * ret_mat[index]->real_col_dim;

        // tmp_mat := mat + s
        //blas_dcopy(real_total_dim, ret_mat[index]->element, 1, tmp_mat[0]->element, 1);
        //cblas_daxpy(real_total_dim, 1.0, s->element, 1, tmp_mat[0]->element, 1);
        cblas_daxpy(real_total_dim, 1.0, s->element, 1, ret_mat[index]->element, 1);

        // high_mat := tmp_mat - s
        cblas_daxpy(real_total_dim, -1.0, s->element, 1, ret_mat[index]->element, 1);

        // low_mat := mat - high_mat
        //cblas_dcopy(real_total_dim, tmp_mat[0]->element, 1, ret_mat[index]->element, 1);
#else // USE_IMKL

        // tmp_mat := mat + s
        add_dmatrix(tmp_mat[0], ret_mat[index], s);

        // high_mat := tmp_mat - s
        sub_dmatrix(tmp_mat[1], tmp_mat[0], s);
        subst_dmatrix(ret_mat[index], tmp_mat[1]);

#endif // USE_IMKL

        // low_mat := mat - high_mat
        sub_mpfmatrix_dmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
        //sub_mpfmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);

        real_num_div = index + 1;
    }

    // free s
	free_dmatrix(s);
    free_dmatrix(tmp_mat[0]);
    free_dmatrix(tmp_mat[1]);
    free_mpfmatrix(tmp_org_mat);

    return real_num_div;
}

// Matrix multiplication based on Ozaki scheme
void mul_mpfmatrix_oz(MPFMatrix ret, MPFMatrix a, int max_num_div_a, MPFMatrix b, int max_num_div_b)
{
    int i, j;
    long int row_dim = ret->row_dim, col_dim = ret->col_dim, mid_dim = a->col_dim;
    long int real_total_dim;
    int real_num_div_a, real_num_div_b;
    DMatrix *div_a, *div_b, div_ret;
    MPFMatrix tmp_ret;

    if(mid_dim != b->row_dim)
    {
        fprintf(stderr, "ERROR: mul_mpfmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
        return;
    }

    //tmp_ret = init2_mpfmatrix(row_dim, col_dim, ret->prec);

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_b = (DMatrix *)calloc(max_num_div_b, sizeof(DMatrix));
    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, mid_dim);
    for(i = 0; i < max_num_div_b; i++)
        div_b[i] = init_dmatrix(mid_dim, col_dim);
    div_ret = init_dmatrix(row_dim, col_dim);

    real_num_div_a = split_mpfmatrix_dmat(div_a, max_num_div_a, a);
    real_num_div_b = split_mpfmatrix_t_dmat(div_b, max_num_div_b, b);

    set0_mpfmatrix(ret);
    for(i = 0; i < real_num_div_a; i++)
    {
        //for(j = 0; j < real_num_div_b; j++)
        for(j = 0; j < real_num_div_b - i; j++)
        {
#ifdef USE_IMKL
            set0_dmatrix(div_ret);
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                div_a[i]->real_row_dim, // m
                div_b[j]->real_col_dim, // n
                div_a[i]->real_col_dim, // k
                1.0,
                div_a[i]->element,
                div_a[i]->real_col_dim, // k
                div_b[j]->element,
                div_b[j]->real_col_dim, // n
                1.0,
                div_ret->element,
                div_ret->real_col_dim   // n
              );
#else // USE_IMKL
            mul_dmatrix(div_ret, div_a[i], div_b[j]);
            //subst_mpfmatrix_dmat(tmp_ret, div_ret);
#endif // USE_IMKL

            add_mpfmatrix_dmat(ret, ret, div_ret);
            //add_mpfmatrix(ret, ret, tmp_ret);
       }
    }

    free_dmatrix(div_ret);
    for(i = 0; i < max_num_div_a; i++)
        free_dmatrix(div_a[i]);
    for(i = 0; i < max_num_div_b; i++)
        free_dmatrix(div_b[i]);

    free(div_a);
    free(div_b);

    //free_mpfmatrix(tmp_ret);

}

// Matrix-Vector multiplication based on Ozaki scheme
void mul_mpfmatrix_mpfvec_oz(MPFVector ret, MPFMatrix a, int max_num_div_a, MPFVector vb, int max_num_div_vb) //, int num_digits)
{
    int i, j;
    int real_num_div_a, real_num_div_vb;
    long int vec_dim = ret->dim, row_dim = a->row_dim, col_dim = a->col_dim;
    DMatrix *div_a;
    DVector *div_vb, div_ret;

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_vb = (DVector *)calloc(max_num_div_vb, sizeof(DVector));

    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, col_dim);
    for(i = 0; i < max_num_div_vb; i++)
        div_vb[i] = init_dvector(vec_dim);
    div_ret = init_dvector(vec_dim);

    real_num_div_a = split_mpfmatrix_dmat(div_a, max_num_div_a, a);
    real_num_div_vb = split_mpfvector_dvec(div_vb, max_num_div_vb, vb);

    set0_mpfvector(ret);
    for(i = 0; i < real_num_div_a; i++)
    {
        for(j = 0; j < real_num_div_vb; j++)
        {

#ifdef USE_IMKL
            set0_dvector(div_ret);
            cblas_dgemv(
                CblasRowMajor,
                CblasNoTrans,
                div_a[i]->real_row_dim,
                div_a[i]->real_col_dim,
                1.0,
                div_a[i]->element,
                div_a[i]->real_row_dim,
                div_vb[j]->element,
                1,
                1.0,
                div_ret->element,
                1
            );
#else // USE_IMKL
            //mul_dmatrix(div_ret, div_a[i], div_vb[j]);
            mul_dmatrix_dvec(div_ret, div_a[i], div_vb[j]); // fixed! 2025-07-09
#endif // USE_IMKL

            add_mpfvector_dvec(ret, ret, div_ret);
       }
    }

    free_dvector(div_ret);
    for(i = 0; i < max_num_div_a; i++)
        free_dmatrix(div_a[i]);
    for(i = 0; i < max_num_div_vb; i++)
        free_dvector(div_vb[i]);

    free(div_a);
    free(div_vb);

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
