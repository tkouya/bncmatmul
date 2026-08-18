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
int split_ddvector_dvec(DVector ret_vec[], int num_div, DDVector org_vec)
{
    long int dim = org_vec->dim;
    int index, num_bits = 53; // IEEE754 binary64
    int real_num_div;
    long int i;
    double org_vec_i, ret_high_vec_i, tmp[DDSIZE];
    double absmax_org_vec, threshold, t_exp; 
    DDVector tmp_org_vec;
    DVector in_ret_vec = NULL;

    if(ret_vec == NULL)
        in_ret_vec = init_dvector(dim);
    else
        in_ret_vec = ret_vec[0];

    // tmp_org_vec := org_vec
    tmp_org_vec = init_ddvector(dim);
    subst_ddvector(tmp_org_vec, org_vec);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        //subst_dvector_ddvec(ret_vec[index], tmp_org_vec);
        //absmax_org_vec = absmax_dvector(NULL, ret_vec[index]);
        if(ret_vec != NULL)
            in_ret_vec = ret_vec[index];

        subst_dvector_ddvec(in_ret_vec, tmp_org_vec);
        absmax_org_vec = absmax_dvector(NULL, in_ret_vec);

        // ret_vec[index] == 0
        if(absmax_org_vec == 0.0) break;

        // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
        //t_exp = ceil(DLOG2(absmax_org_vec)) + ceil(((double)num_bits + DLOG2((double)(dim + 1))) / 2.0);
        t_exp = ceil(DLOG2(absmax_org_vec)) + ceil(((double)num_bits + DLOG2((double)(dim))) / 2.0);

        //rdd_pow(threshold, two, t_exp);
        //rdd_pow_mpfr(threshold, two, t_exp);
        threshold = pow(2.0, t_exp);

        for(i = 0; i < dim; i++)
        {
            // set high vector
            //rdd_set(org_vec_i, get_ddvector_i(tmp_org_vec, i)); 
            //org_vec_i = get_dvector_i(ret_vec[index], i);   
            org_vec_i = get_dvector_i(in_ret_vec, i);  
            ret_high_vec_i = org_vec_i + threshold;
            //rdd_add(ret_high_vec_i, org_vec_i, threshold);
            ret_high_vec_i -= threshold;
            //rdd_sub(ret_high_vec_i, ret_high_vec_i, threshold);
            //set_dvector_i(ret_vec[index], i, ret_high_vec_i);
            set_dvector_i(in_ret_vec, i, ret_high_vec_i);

            // set low vector
            rdd_sub_d(tmp, get_ddvector_i(tmp_org_vec, i), ret_high_vec_i);           
            set_ddvector_i(tmp_org_vec, i, tmp);
        }

        real_num_div = index + 1;

    }

    free_ddvector(tmp_org_vec);
    if(ret_vec == NULL)
        free_dvector(in_ret_vec);

    return real_num_div;
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
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;
    double bij[DDSIZE];

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_ddmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_ddmatrix_dmat\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

	double tmp[DDSIZE];
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//rdd_add_d(tmp, get_ddmatrix_ij(a, i, j), get_dmatrix_ij(b, i, j));
            bij[0] = get_dmatrix_ij(b, i, j); bij[1] = 0.0;
            rdd_add(tmp, get_ddmatrix_ij(a, i, j), bij);
			set_ddmatrix_ij(c, i, j, tmp);
		}
	}
}

/* c := a - (double)b */
void sub_ddmatrix_dmat(DDMatrix c, DDMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;
    double bij[DDSIZE];

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_ddmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_ddmatrix_dmat\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

	double tmp[DDSIZE];
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//rdd_sub_d(tmp, get_ddmatrix_ij(a, i, j), get_dmatrix_ij(b, i, j));
            bij[0] = get_dmatrix_ij(b, i, j); bij[1] = 0.0;
            rdd_sub(tmp, get_ddmatrix_ij(a, i, j), bij);
			set_ddmatrix_ij(c, i, j, tmp);
		}
	}
}

//#define SPLIT_NUM_DIGITS 53
//#define SPLIT_NUM_DIGITS 55
//#define SPLIT_NUM_DIGITS 56
//#define SPLIT_NUM_DIGITS 57
//#define SPLIT_NUM_DIGITS 64

// SplitMat_A
// return real_num_div
int split_ddmatrix_dmat(DMatrix ret_mat[], int num_div, DDMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, real_total_dim;
	long int num_digits = 53; // IEEE double prec.
    int real_num_div;
	//long int num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//long int num_digits = 64; // IEEE double prec.
	//double *s;
    DDMatrix tmp_org_mat;
    DMatrix s, tmp_mat[2], in_ret_mat;
	double mu, mu_total, abs_aij, t_exp, power2;

    row_dim = org_mat->row_dim;
    //row_dim = org_mat->real_row_dim;
    col_dim = org_mat->col_dim;
    real_total_dim = org_mat->real_row_dim * org_mat->real_col_dim;

	// threshold matrix 
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    //s = init_ddmatrix(row_dim, col_dim);
    s = init_dmatrix(row_dim, col_dim);

    //tmp_mat[0] = init_ddmatrix(row_dim, col_dim);
    //tmp_mat[1] = init_ddmatrix(row_dim, col_dim);
    tmp_mat[0] = init_dmatrix(row_dim, col_dim);
    tmp_mat[1] = init_dmatrix(row_dim, col_dim);
    if(ret_mat == NULL)
        in_ret_mat = init_dmatrix(row_dim, col_dim);
    else
        in_ret_mat = ret_mat[0];

    // tmp_org_mat := org_mat;
    tmp_org_mat = init_ddmatrix(row_dim, col_dim);
    subst_ddmatrix(tmp_org_mat, org_mat);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        //printf("In split_ddmatrix ... index= %d\n", index);
        //set0_dmatrix(ret_mat[index]);
        if(ret_mat != NULL)
            in_ret_mat = ret_mat[index];

        //subst_dmatrix_ddmat(ret_mat[index], tmp_org_mat);
        subst_dmatrix_ddmat(in_ret_mat, tmp_org_mat);

        // mu[i] = max_j |mat[i, j]|
        // mu_total = sum mu
        mu_total = 0.0;

        for(i = 0; i < row_dim; i++)
        {
            //absmax_row_ddmatrix(mu, NULL, i, tmp_org_mat);
            //mu = absmax_row_dmatrix(NULL, i, ret_mat[index]);
            mu = absmax_row_dmatrix(NULL, i, in_ret_mat);

            mu_total += mu;

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            //t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);
            //t_exp[1] = 0.0;
            //printf("     num, num_digits, col_dim + 1   = %15.7e, %ld, %ld\n", mu, num_digits, col_dim + 1);
            //printf("log2(num, num_digits + col_dim + 1) = %15.7e, %15.7e\n", DLOG2(mu), ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0));
            //t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);
            t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(col_dim))) / 2.0);

            // s[i, j] = 2^t_exp
            //rdd_pow(power2, two, t_exp);
            //rdd_pow_mpfr(power2, two, t_exp);
            power2 = pow(2.0, t_exp);
            //printf("power2 = %15.7e\n", power2);
            //printf("index, i, power2 = %ld, %ld, %25.17e\n", index, i, power2[0]);
            for(j = 0; j < col_dim; j++)
            {
                //s[i * col_dim + j] = pow(2.0, t_exp);
                //set_ddmatrix_ij(s, i, j, power2);
                set_dmatrix_ij(s, i, j, power2);
            }
        }

        // if ret_mat[index] == 0 -> break
        if(mu_total == 0.0) break;

        // split org_mat to ret_high_mat and ret_low_mat
#ifdef USE_IMKL
        //real_total_dim = ret_mat[index]->real_row_dim * ret_mat[index]->real_col_dim;
        real_total_dim = in_ret_mat->real_row_dim * in_ret_mat->real_col_dim;

        // tmp_mat := mat + s
        //blas_dcopy(real_total_dim, ret_mat[index]->element, 1, tmp_mat[0]->element, 1);
        //cblas_daxpy(real_total_dim, 1.0, s->element, 1, tmp_mat[0]->element, 1);
        //cblas_daxpy(real_total_dim, 1.0, s->element, 1, ret_mat[index]->element, 1);
        cblas_daxpy(real_total_dim, 1.0, s->element, 1, in_ret_mat->element, 1);

        // high_mat := tmp_mat - s
        //cblas_daxpy(real_total_dim, -1.0, s->element, 1, ret_mat[index]->element, 1);
        cblas_daxpy(real_total_dim, -1.0, s->element, 1, in_ret_mat->element, 1);

        // low_mat := mat - high_mat
        //cblas_dcopy(real_total_dim, tmp_mat[0]->element, 1, ret_mat[index]->element, 1);
#else // USE_IMKL
        // tmp_mat := mat + s
        //add_dmatrix(tmp_mat[0], ret_mat[index], s);
        add_dmatrix(tmp_mat[0], in_ret_mat, s);

        // high_mat := tmp_mat - s
        sub_dmatrix(tmp_mat[1], tmp_mat[0], s);
        //subst_dmatrix(ret_mat[index], tmp_mat[1]);
        subst_dmatrix(in_ret_mat, tmp_mat[1]);

#endif // USE_IMKL

        // low_mat := mat - high_mat
        //sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);
        //sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
        sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, in_ret_mat);

        // real_num_div = index + 1;
        real_num_div = index + 1;
    }

	// free s
	//free_ddmatrix(s);
	free_dmatrix(s);
    free_ddmatrix(tmp_org_mat);
    free_dmatrix(tmp_mat[0]);
    free_dmatrix(tmp_mat[1]);
    if(ret_mat == NULL)
        free_dmatrix(in_ret_mat);

    return real_num_div;
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

// SplitMat_B
// return real_num_div
int split_ddmatrix_t_dmat(DMatrix ret_mat[], int num_div, DDMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, real_total_dim;
	int real_num_div, num_digits = 53; // IEEE double prec.
	//int flag_stop = 0; //, num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//int flag_stop = 0, num_digits = 64; // IEEE double prec.
	//double *s;
    DDMatrix tmp_org_mat;
    DMatrix s, tmp_mat[2], in_ret_mat;
	//double mu[DDSIZE], abs_aij[DDSIZE], t_exp[DDSIZE], power2[DDSIZE], two[DDSIZE] = {2.0, 0.0};
	double mu, abs_aij, t_exp, power2, mu_total;

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;
    real_total_dim = org_mat->real_row_dim * org_mat->real_col_dim;

	// initialize s
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    //s = init_ddmatrix(row_dim, col_dim);
    s = init_dmatrix(row_dim, col_dim);

    tmp_mat[0] = init_dmatrix(row_dim, col_dim);
    tmp_mat[1] = init_dmatrix(row_dim, col_dim);
    if(ret_mat == NULL)
        in_ret_mat = init_dmatrix(row_dim, col_dim);

    tmp_org_mat = init_ddmatrix(row_dim, col_dim);
    subst_ddmatrix(tmp_org_mat, org_mat);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        if(ret_mat != NULL)
            in_ret_mat = ret_mat[index];

        //subst_dmatrix_ddmat(ret_mat[index], tmp_org_mat);
        subst_dmatrix_ddmat(in_ret_mat, tmp_org_mat);

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
            //mu = absmax_col_dmatrix(NULL, j, ret_mat[index]);
            mu = absmax_col_dmatrix(NULL, j, in_ret_mat);

            mu_total += mu;
            //printf("mu%d: %15.7e ", j, mu);

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            //t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);
            //t_exp[1] = 0.0;
            t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);
            //if(isnan(t_exp))
            //    flag_stop = 1;

            // s[i, j] = 2^t_exp
            //rdd_pow(power2, two, t_exp);
            //rdd_pow_mpfr(power2, two, t_exp);
            power2 = pow(2.0, t_exp);

            //printf("index, j, power2 = %ld, %ld, %25.17e\n", index, j, power2[0]);
            for(i = 0; i < row_dim; i++)
                set_dmatrix_ij(s, i, j, power2);
                //s[i * col_dim + j] = pow(2.0, t_exp);
        }
        //if(flag_stop == 1)
        //    break;
        if(mu_total == 0.0) break;

#ifdef USE_IMKL
        //real_total_dim = ret_mat[index]->real_row_dim * ret_mat[index]->real_col_dim;
        real_total_dim = in_ret_mat->real_row_dim * in_ret_mat->real_col_dim;

        // tmp_mat := mat + s
        //blas_dcopy(real_total_dim, ret_mat[index]->element, 1, tmp_mat[0]->element, 1);
        //cblas_daxpy(real_total_dim, 1.0, s->element, 1, tmp_mat[0]->element, 1);
        //cblas_daxpy(real_total_dim, 1.0, s->element, 1, ret_mat[index]->element, 1);
        cblas_daxpy(real_total_dim, 1.0, s->element, 1, in_ret_mat->element, 1);

        // high_mat := tmp_mat - s
        //cblas_daxpy(real_total_dim, -1.0, s->element, 1, ret_mat[index]->element, 1);
        cblas_daxpy(real_total_dim, -1.0, s->element, 1, in_ret_mat->element, 1);

        // low_mat := mat - high_mat
        //cblas_dcopy(real_total_dim, tmp_mat[0]->element, 1, ret_mat[index]->element, 1);
#else // USE_IMKL
        // tmp_mat := mat + s
        //add_dmatrix(tmp_mat[0], ret_mat[index], s);
        add_dmatrix(tmp_mat[0], in_ret_mat, s);

        // high_mat := tmp_mat - s
        sub_dmatrix(tmp_mat[1], tmp_mat[0], s);
        //subst_dmatrix(ret_mat[index], tmp_mat[1]);
        subst_dmatrix(in_ret_mat, tmp_mat[1]);

#endif // USE_IMKL

        // low_mat := mat - high_mat
        //sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
        sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, in_ret_mat);
        //sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);

        real_num_div = index + 1;
    }

    // free s
	free_dmatrix(s);
    free_dmatrix(tmp_mat[0]);
    free_dmatrix(tmp_mat[1]);
    free_ddmatrix(tmp_org_mat);
    if(ret_mat == NULL)
        free_dmatrix(in_ret_mat);

    return real_num_div;
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

// Matrix multiplication based on Ozaki scheme
void mul_ddmatrix_oz(DDMatrix ret, DDMatrix a, int max_num_div_a, DDMatrix b, int max_num_div_b) //, int num_digits)
{
    int i, j;
    int real_num_div_a, real_num_div_b;
    long int row_dim = ret->row_dim, col_dim = ret->col_dim, mid_dim = a->col_dim;
    DMatrix *div_a, *div_b, div_ret;

    if(mid_dim != b->row_dim)
    {
        fprintf(stderr, "ERROR: mul_ddmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
        return;
    }

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_b = (DMatrix *)calloc(max_num_div_b, sizeof(DMatrix));
    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, mid_dim);
    for(i = 0; i < max_num_div_b; i++)
        div_b[i] = init_dmatrix(mid_dim, col_dim);
    div_ret = init_dmatrix(row_dim, col_dim);

    real_num_div_a = split_ddmatrix_dmat(div_a, max_num_div_a, a);
    //printf("split_ddmatrix_dmat(%d, %d)  ->%d\n", div_a[0]->real_row_dim, div_a[0]->real_col_dim, real_num_div_a);
    real_num_div_b = split_ddmatrix_t_dmat(div_b, max_num_div_b, b);
    //printf("split_ddmatrix_t_dmat(%d, %d)->%d\n", div_b[0]->real_row_dim, div_b[0]->real_col_dim, real_num_div_b);

    set0_ddmatrix(ret);
    for(i = 0; i < real_num_div_a; i++)
    {
        //for(j = 0; j < real_num_div_b; j++)
        for(j = 0; j < real_num_div_b - i; j++)
        {

            //rintf("(i, j) = (%d, %d), %d, %d\n", i, j, div_b[j]->real_row_dim, div_b[j]->real_col_dim);
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
#endif // USE_IMKL

            add_ddmatrix_dmat(ret, ret, div_ret);
       }
    }

    free_dmatrix(div_ret);
    for(i = 0; i < max_num_div_a; i++)
        free_dmatrix(div_a[i]);
    for(i = 0; i < max_num_div_b; i++)
        free_dmatrix(div_b[i]);

    free(div_a);
    free(div_b);

}

// Matrix-Vector multiplication based on Ozaki scheme
void mul_ddmatrix_ddvec_oz(DDVector ret, DDMatrix a, int max_num_div_a, DDVector vb, int max_num_div_vb) //, int num_digits)
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

    real_num_div_a = split_ddmatrix_dmat(div_a, max_num_div_a, a);
    real_num_div_vb = split_ddvector_dvec(div_vb, max_num_div_vb, vb);

    set0_ddvector(ret);
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
            mul_dmatrix_dvec(div_ret, div_a[i], div_vb[j]); // Fix! 2024-07-30 T.Kouya
#endif // USE_IMKL

            add_ddvector_dvec(ret, ret, div_ret);
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
