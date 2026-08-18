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
int split_tdvector_dvec(DVector ret_vec[], int num_div, TDVector org_vec) //, int num_bits)
{
    long int dim = org_vec->dim;
    int index, real_num_div, num_bits = 53; // IEEE754 binary64
    long int i;
    double org_vec_i, ret_high_vec_i, tmp[TDSIZE];
    double absmax_org_vec, threshold, t_exp; 
    TDVector tmp_org_vec;

    // tmp_org_vec := org_vec
    tmp_org_vec = init_tdvector(dim);
    subst_tdvector(tmp_org_vec, org_vec);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        subst_dvector_tdvec(ret_vec[index], tmp_org_vec);
        absmax_org_vec = absmax_dvector(NULL, ret_vec[index]);

        // ret_vec[index] == 0
        if(absmax_org_vec == 0.0) break;
        // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
        //t_exp = ceil(DLOG2(absmax_org_vec)) + ceil(((double)num_bits + DLOG2((double)(dim + 1))) / 2.0);
        t_exp = ceil(DLOG2(absmax_org_vec)) + ceil(((double)num_bits + DLOG2((double)(dim))) / 2.0);

        //rtd_pow(threshold, two, t_exp);
        //rtd_pow_mpfr(threshold, two, t_exp);
        threshold = pow(2.0, t_exp);

        for(i = 0; i < dim; i++)
        {
            // set high vector
            //rtd_set(org_vec_i, get_tdvector_i(tmp_org_vec, i)); 
            org_vec_i = get_dvector_i(ret_vec[index], i);   
            ret_high_vec_i = org_vec_i + threshold;
            //rtd_add(ret_high_vec_i, org_vec_i, threshold);
            ret_high_vec_i -= threshold;
            //rtd_sub(ret_high_vec_i, ret_high_vec_i, threshold);
            set_dvector_i(ret_vec[index], i, ret_high_vec_i);

            // set low vector
            rtd_sub_d(tmp, get_tdvector_i(tmp_org_vec, i), ret_high_vec_i);           
            set_tdvector_i(tmp_org_vec, i, tmp);
        }

        real_num_div = index + 1;
    }

    free_tdvector(tmp_org_vec);

    return real_num_div;
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
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_tdmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_tdmatrix_dmat\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

	double tmp[TDSIZE], bij[TDSIZE];
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//rtd_add_d(tmp, get_tdmatrix_ij(a, i, j), get_dmatrix_ij(b, i, j));
            bij[0] = get_dmatrix_ij(b, i, j); bij[1] = 0.0, bij[2] = 0.0;
            rtd_add(tmp, get_tdmatrix_ij(a, i, j), bij);
			set_tdmatrix_ij(c, i, j, tmp);
		}
	}
}

/* c := a - (doble)b */
void sub_tdmatrix_dmat(TDMatrix c, TDMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_tdmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_tdmatrix_dmat\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

	double tmp[TDSIZE], bij[TDSIZE];
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//rtd_sub_d(tmp, get_tdmatrix_ij(a, i, j), get_dmatrix_ij(b, i, j));
            bij[0] = get_dmatrix_ij(b, i, j); bij[1] = 0.0, bij[2] = 0.0;
            rtd_sub(tmp, get_tdmatrix_ij(a, i, j), bij);
			set_tdmatrix_ij(c, i, j, tmp);
		}
	}
}

//#define SPLIT_NUM_DIGITS 64

// SplitMat_A
int split_tdmatrix_dmat(DMatrix ret_mat[], int num_div, TDMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, real_total_dim;
	long int num_digits = 53; // IEEE double prec.
	//long int num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//double *s;
    int real_num_div;
    TDMatrix tmp_org_mat;
    DMatrix s, tmp_mat[2]; 
	double mu, abs_aij, t_exp, power2, mu_total;

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;

	// threshold matrix 
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    //s = init_tdmatrix(row_dim, col_dim);
    s = init_dmatrix(row_dim, col_dim);

    //tmp_mat[0] = init_tdmatrix(row_dim, col_dim);
    //tmp_mat[1] = init_tdmatrix(row_dim, col_dim);
    tmp_mat[0] = init_dmatrix(row_dim, col_dim);
    tmp_mat[1] = init_dmatrix(row_dim, col_dim);

    // tmp_org_mat := org_mat;
    tmp_org_mat = init_tdmatrix(row_dim, col_dim);
    subst_tdmatrix(tmp_org_mat, org_mat);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        //printf("In split_tdmatrix ... index= %d\n", index);
        //set0_dmatrix(ret_mat[index]);
        subst_dmatrix_tdmat(ret_mat[index], tmp_org_mat);

        // mu[i] = max_j |mat[i, j]|
        // mu_total += mu
        mu_total = 0.0;
        for(i = 0; i < row_dim; i++)
        {
            //absmax_row_tdmatrix(mu, NULL, i, tmp_org_mat);
            mu = absmax_row_dmatrix(NULL, i, ret_mat[index]);
            mu_total += mu;

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            //t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);
            //t_exp[1] = 0.0;
            t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(col_dim))) / 2.0);

            // s[i, j] = 2^t_exp
            //rtd_pow(power2, two, t_exp);
            //rtd_pow_mpfr(power2, two, t_exp);
            power2 = pow(2.0, t_exp);
            //printf("index, i, power2 = %ld, %ld, %25.17e\n", index, i, power2[0]);
            for(j = 0; j < col_dim; j++)
            {
                //s[i * col_dim + j] = pow(2.0, t_exp);
                //set_tdmatrix_ij(s, i, j, power2);
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
        //add_tdmatrix(tmp_mat[0], tmp_org_mat, s);
        add_dmatrix(tmp_mat[0], ret_mat[index], s);

        // high_mat := tmp_mat - s
        //sub_tdmatrix(tmp_mat[1], tmp_mat[0], s);
        sub_dmatrix(tmp_mat[1], tmp_mat[0], s);
        subst_dmatrix(ret_mat[index], tmp_mat[1]);
#endif // USE_IMKL

        // low_mat := mat - high_mat
        sub_tdmatrix_dmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
        //sub_tdmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);

        real_num_div = index + 1;
    }

	// free s
	//free_tdmatrix(s);
	free_dmatrix(s);
    free_tdmatrix(tmp_org_mat);
    free_dmatrix(tmp_mat[0]);
    free_dmatrix(tmp_mat[1]);

    return real_num_div;
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
int split_tdmatrix_t_dmat(DMatrix ret_mat[], int num_div, TDMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, real_total_dim;
	int real_num_div, num_digits = 53; // IEEE double prec.
	//int flag_stop = 0, num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//double *s;
    TDMatrix tmp_org_mat;
    DMatrix s, tmp_mat[2];
	//double mu[TDSIZE], abs_aij[TDSIZE], t_exp[TDSIZE], power2[TDSIZE], two[TDSIZE] = {2.0, 0.0};
	double mu, abs_aij, t_exp, power2, mu_total;

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;

	// initialize s
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    //s = init_tdmatrix(row_dim, col_dim);
    s = init_dmatrix(row_dim, col_dim);

    tmp_mat[0] = init_dmatrix(row_dim, col_dim);
    tmp_mat[1] = init_dmatrix(row_dim, col_dim);

    tmp_org_mat = init_tdmatrix(row_dim, col_dim);
    subst_tdmatrix(tmp_org_mat, org_mat);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        subst_dmatrix_tdmat(ret_mat[index], tmp_org_mat);

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
            //rtd_pow(power2, two, t_exp);
            //rtd_pow_mpfr(power2, two, t_exp);
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
        sub_tdmatrix_dmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
        //sub_tdmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);

        real_num_div = index + 1;
    }

    // free s
	free_dmatrix(s);
    free_dmatrix(tmp_mat[0]);
    free_dmatrix(tmp_mat[1]);
    free_tdmatrix(tmp_org_mat);

    return real_num_div;
}

// Matrix multiplication based on Ozaki scheme
void mul_tdmatrix_oz(TDMatrix ret, TDMatrix a, int max_num_div_a, TDMatrix b, int max_num_div_b)
{
    int i, j;
    long int row_dim = ret->row_dim, col_dim = ret->col_dim, mid_dim = a->col_dim;
    int real_num_div_a, real_num_div_b;
    DMatrix *div_a, *div_b, div_ret;
    TDMatrix tmp_ret;

    if(mid_dim != b->row_dim)
    {
        fprintf(stderr, "ERROR: mul_tdmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
        return;
    }

    tmp_ret = init_tdmatrix(row_dim, col_dim);

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_b = (DMatrix *)calloc(max_num_div_b, sizeof(DMatrix));
    if(div_a == NULL) 
    {
        fprintf(stderr, "ERROR: div_a(max_num_div_a = %d) is null in mul_tdmatrix_oz!\n", max_num_div_a);
        return;
    }
    if(div_b == NULL) 
    {
        fprintf(stderr, "ERROR: div_b(max_num_div_b = %d) is null in mul_tdmatrix_oz!\n", max_num_div_b);
        return;
    }
    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, mid_dim);
    for(i = 0; i < max_num_div_b; i++)
        div_b[i] = init_dmatrix(mid_dim, col_dim);
    div_ret = init_dmatrix(row_dim, col_dim);

    real_num_div_a = split_tdmatrix_dmat(div_a, max_num_div_a, a);
    real_num_div_b = split_tdmatrix_t_dmat(div_b, max_num_div_b, b);
    //printf("Split done!\n");

    set0_tdmatrix(ret);
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
#endif // USE_IMKL
            //printf("DMul done!\n");
            //subst_tdmatrix_dmat(tmp_ret, div_ret);
            add_tdmatrix_dmat(ret, ret, div_ret);
            //add_tdmatrix(ret, ret, tmp_ret);
            //printf("Add done!\n");
       }
    }

    free_dmatrix(div_ret);
    for(i = 0; i < max_num_div_a; i++)
        free_dmatrix(div_a[i]);
    for(i = 0; i < max_num_div_b; i++)
        free_dmatrix(div_b[i]);

    free(div_a);
    free(div_b);

    free_tdmatrix(tmp_ret);

}

// Matrix-Vector multiplication based on Ozaki scheme
void mul_tdmatrix_tdvec_oz(TDVector ret, TDMatrix a, int max_num_div_a, TDVector vb, int max_num_div_vb) //, int num_digits)
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

    real_num_div_a = split_tdmatrix_dmat(div_a, max_num_div_a, a);
    real_num_div_vb = split_tdvector_dvec(div_vb, max_num_div_vb, vb);

    set0_tdvector(ret);
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

            add_tdvector_dvec(ret, ret, div_ret);
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
