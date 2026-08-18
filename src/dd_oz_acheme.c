/********************************************************************************/
/* dd_oz_scheme: Multiple precision linear computation based on Ozaki scheme.      */
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
#define DLOG2(x) (log10((x)) / 0.30102999566398119521373889472449)

// absmax_ddvector
void absmax_ddvector(double ret[DDSIZE], long int *max_index, DDVector vec)
{
    long int i, max_i, dim = vec->dim;
    double abs_val[DDSIZE];

    max_i = 0;
    for(i = 0; i < dim; i++)
    {
        //abs_val = fabs(get_ddvector_i(vec, i));
        rdd_abs(abs_val, get_ddmatrix_ij(vec, i));
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

// extract vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
void extract_ddvector(DVector ret_vec[], ing num_div, DDVector org_vec, double num_bits)
{
    long int dim = org_vec->dim;
    int index;
    long int i;
    double org_vec_i[DDSIZE], ret_high_vec_i[DDSIZE], absmax_org_vec[DDSIZE], tmp[DDSIZE], ;
    double threshold[DDSIZE], t_exp[DDSIZE], two[DDSIZE] = {2.0, 0.0};
    DDVector tmp_org_vec;

    // tmp_org_vec := org_vec
    tmp_org_vec = init_ddvector(dim);
    subst_ddvector(tmp_org_vec, org_vec);

	absmax_ddvector(absmax_org_vec, NULL, org_vec);

    // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
	t_exp[0] = ceil(DLOG2(absmax_org_vec[0])) + ceil(((double)num_bits + DLOG2((double)(dim + 1))) / 2.0);
    t_exp[1] = 0.0;
    
    threshold = rdd_pow(two, t_exp);

    for(index = 0; index < num_div; index++)
    {
        set0_dvector(ret_vec[index]);
        for(i = 0; i < dim; i++)
        {
            // set high vector
            rdd_set(org_vec_i, get_ddvector_i(tmp_org_vec, i));
            //ret_high_vec_i = org_vec_i + threshold;
            rdd_add(ret_high_vec_i, org_vec_i, threshold);
            //ret_high_vec_i -= threshold;
            rdd_sub(ret_high_vec_i, ret_high_vec_i, threshold);
            set_dvector_i(ret_high_vec, i, ret_high_vec_i[0]);

            // set low vector
            rdd_sub_d(tmp, org_vec_i, ret_high_vec_i[0]);
            set_ddvector_i(tmp_org_vec, i, tmp);
        }
    }

    free_ddvector(tmp_org_vec);
}

// absmax_row_ddmatrix
void absmax_row_ddmatrix(dboule mu[DDSIZE], long int *max_j, long int row_index, DDMatrix mat)
{
    long int j, max_index = 0;
    double abs_aij[DDSIZE];

	//mu = fabs(mat[i * col_dim + 0]);
    rdd_abs(mu, get_ddmatrix_ij(mat, row_index, 0));

	for(j = 1; j < mat->col_dim; j++)
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

// SplitMat_A
void split_ddmatrix(DMatrix ret_mat[], int num_div, DDMatrix org_mat)
{
	long int i, index, row_dim, col_dim;
	long int num_digits = 53; // IEEE double prec.
	//double *s;
    DDMatrix s, tmp_mat[2];
	double mu[DDSIZE], abs_aij[DDSIZE], t_exp[DDSIZE], power2[DDSIZE], two[DDSIZE] = {2.0, 0.0};

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;

	// threshold matrix 
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    s = init_ddmatrix(row_dim, col_dim);

    tmp_mat[0] = init_ddmatrix(row_dim, col_dim);
    tmp_mat[1] = init_ddmatrix(row_dim, col_dim);

    // tmp_org_mat := org_mat;
    tmp_org_mat = init_ddmatrix(row_dim, col_dim);
    subst_ddmatrix(tmp_org_mat, org_mat);

    for(index = 0; index < num_dim; i++)
    {
        set0_ddmatrix(ret_mat[index]);

        // mu[i] = max_j |mat[i, j]|
        for(i = 0; i < row_dim; i++)
        {
            mu = absmax_row_ddmatrix(NULL, i, tmp_org_mat);

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);
            t_exp[1] = 0.0;

            // s[i, j] = 2^t_exp
            rdd_pow(power2, two, t_exp);
            for(j = 0; j < col_dim; j++)
            {
                //s[i * col_dim + j] = pow(2.0, t_exp);
                set_ddmatrix_ij(s, i, j, power2);
            }
        }

        // split org_mat to ret_high_mat and ret_low_mat

        // tmp_mat := mat + s
        add_ddmatrix(tmp_mat[0], tmp_org_mat, s);

        // high_mat := tmp_mat - s
        sub_ddmatrix(tmp_mat[1], tmp_mat[0], s);
        subst_dmatrix_ddmat(ret_mat[index], tmp_mat[1]);

        // low_mat := mat - high_mat
        sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
    }

	// free s
	free_ddmatrix(s);
    free_ddmatrix(tmp_org_mat);
    free_ddmatrix(tmp_mat[0]);
    free_ddmatrix(tmp_mat[1]);
}

// absmax_col_ddmatrix
void absmax_col_ddmatrix(mu[DDSIZE], long int *max_i, long int col_index, DDMatrix mat)
{
    long int i, max_index = 0;
    double abs_aij[DDSIZE];

	//mu = fabs(mat[0 * col_dim + j]);
    rdd_abs(mu, get_ddmatrix_ij(mat, 0, col_index));
	for(i = 1; i < mat->row_dim; i++)
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
void split_ddmatrix_t(DMatrix ret_mat[], int num_div, DDMatrix org_mat)
{
	long int index, j, row_dim, col_dim;
	int num_digits = 53; // IEEE double prec.
	//double *s;
    DDMatrix s, tmp_mat[2], tmp_org_mat;
	double mu[DDSIZE], abs_aij[DDSIZE], t_exp[DDSIZE], power2[DDSIZE], two[DDSIZE] = {2.0, 0.0};

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;

	// initialize s
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    s = init_ddmatrix(row_dim, col_dim);

    tmp_mat[0] = init_ddmatrix(row_dim, col_dim);
    tmp_mat[1] = init_ddmatrix(row_dim, col_dim);

    tmp_org_mat = init_ddmatrix(row_dim, col_dim);
    subst_ddmatrix(tmp_org_mat, org_mat);

    for(index = 0; index < num_dim; index++)
    {
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
            mu = absmax_col_ddmatrix(NULL, j, tmp_org_mat);

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            t_exp[0] = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);
            t_exp[1] = 0.0;

            // s[i, j] = 2^t_exp
            rdd_pow(power2, two, t_exp);
            for(i = 0; i < row_dim; i++)
                set_ddmatrix_ij(s, i, j, power2);
                //s[i * col_dim + j] = pow(2.0, t_exp);
        }

        // tmp_mat := mat + s
        add_ddmatrix(tmp_mat[0], tmp_org_mat, s);

        // high_mat := tmp_mat - s
        sub_ddmatrix(tmp_mat[1], tmp_mat[0], s);
        subst_dmatrix_ddmat(ret_mat[index], tmp_mat[1]);

        // low_mat := mat - high_mat
        sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
    }

    // free s
	free_ddmatrix(s);
    free_ddmatrix(tmp_mat[0]);
    free_ddmatrix(tmp_mat[1]);
    free_ddmatrix(tmp_org_mat);
}

#define DIM 5
//#define DIM 10

#define MAX_NUM_DIV 10

int main()
{
    long int i, j;
    double tmp[DDSIZE];
    DDVector vec_org, vec;
    DVector dvec[MAX_NUM_DIV];
    DDMatrix mat_org, mat;
    DMatrix dmat[MAX_NUM_DIV];

// DVector 

    vec_org = init_ddvector(DIM);
    vec = init_ddvector(DIM);
    for(i = 0; i < 10; i++)
        dvec[i] = init_dvector(DIM);

    // set random
    srand(DIM);
    for(i = 0; i < DIM; i++)
    {
        rdd_set_d(tmp, (double)rand());
        rdd_mul_d(tmp, tmp, (double)rand());
        set_ddvector_i(vec_org, i, tmp);
    }

    // split
    extract_ddvector(dvec, MAX_NUM_DIV, vec_org, 53);

    for(i = 0; i < MAX_NUM_DIV; i++)
    {
        printf("dvec[%d]:\n", i);
        print_dvector(dvec[i]);

        add_ddvector_dvec(vec, vec, dvec[i]);
    }
    sub_ddvector(vec, vec, vec_org);
    printf("vec - vec_org:\n"); print_ddvector(vec); 

    free_ddvector(vec_org);
    free_ddvector(vec);
    for(i = 0; i < 10; i++)
        free_dvector(dvec[i]);

// DMatrix

    dmat_org = init_ddmatrix(DIM, DIM);
    dmat = init_ddmatrix(DIM, DIM);
    for(i = 0; i < 10; i++)
        dmat[i] = init_dmatrix(DIM, DIM);

    // set random
    srand(DIM);
    for(i = 0; i < DIM; i++)
    {
        for(j = 0; j < DIM; j++)
        {
            rdd_set_d(tmp, (double)rand());
            rdd_mul_d(tmp, tmp, (double)rand());
            set_ddmatrix_ij(dmat_org, i, j, tmp);
        }
    }

    // split
    split_ddmatrix(dmat, MAX_NUM_DIV, mat_org);

    for(i = 0; i < MAX_NUM_DIV; i++)
    {
        printf("dmat[i]:\n"); print_dmatrix(dmat[i]);

        add_ddmatrix_dmat(mat, mat, dmat[i]);
    }
    sub_dmatrix(mat, mat, mat_org);
    printf("mat - mat_org:\n"); print_ddmatrix(mat); 

    free_ddmatrix(mat_org);
    free_ddmatrix(mat);
    for(i = 0; i < 10; i++)
        free_dmatrix(dmat[i]);


    return 0;
}