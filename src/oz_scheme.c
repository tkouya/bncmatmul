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

// log2(x) := log10(x) / log10(2)
//#define DLOG2(x) (log10((x)) / 0.30102999566398119521373889472449)

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
int split_dvector_dvec(DVector ret_vec[], int num_div, DVector org_vec)
{
    long int dim = org_vec->dim;
    int index, num_bits = 53; // IEEE754 binary64
    int real_num_div;
    long int i;
    double org_vec_i, ret_high_vec_i, tmp;
    double absmax_org_vec, threshold, t_exp; 
    DVector tmp_org_vec;
    DVector in_ret_vec = NULL;

    if(ret_vec == NULL)
        in_ret_vec = init_dvector(dim);
    else
        in_ret_vec = ret_vec[0];

    // tmp_org_vec := org_vec
    tmp_org_vec = init_dvector(dim);
    subst_dvector(tmp_org_vec, org_vec);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        //subst_dvector_ddvec(ret_vec[index], tmp_org_vec);
        //absmax_org_vec = absmax_dvector(NULL, ret_vec[index]);
        if(ret_vec != NULL)
            in_ret_vec = ret_vec[index];

        subst_dvector(in_ret_vec, tmp_org_vec);
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
            //rdd_sub_d(tmp, get_ddvector_i(tmp_org_vec, i), ret_high_vec_i);
            tmp = get_dvector_i(tmp_org_vec, i) - ret_high_vec_i;  
            set_dvector_i(tmp_org_vec, i, tmp);
        }

        real_num_div = index + 1;

    }

    free_dvector(tmp_org_vec);
    if(ret_vec == NULL)
        free_dvector(in_ret_vec);

    return real_num_div;
}

// SplitMat_A
// return real_num_div
int split_dmatrix_dmat(DMatrix ret_mat[], int num_div, DMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, real_total_dim;
	long int num_digits = 53; // IEEE double prec.
    int real_num_div;
	//long int num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//long int num_digits = 64; // IEEE double prec.
	//double *s;
    DMatrix tmp_org_mat;
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
    tmp_org_mat = init_dmatrix(row_dim, col_dim);
    subst_dmatrix(tmp_org_mat, org_mat);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        //printf("In split_ddmatrix ... index= %d\n", index);
        //set0_dmatrix(ret_mat[index]);
        if(ret_mat != NULL)
            in_ret_mat = ret_mat[index];

        //subst_dmatrix_ddmat(ret_mat[index], tmp_org_mat);
        subst_dmatrix(in_ret_mat, tmp_org_mat);

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
        sub_dmatrix(tmp_org_mat, tmp_org_mat, in_ret_mat);

        // real_num_div = index + 1;
        real_num_div = index + 1;
    }

	// free s
	//free_ddmatrix(s);
	free_dmatrix(s);
    free_dmatrix(tmp_org_mat);
    free_dmatrix(tmp_mat[0]);
    free_dmatrix(tmp_mat[1]);
    if(ret_mat == NULL)
        free_dmatrix(in_ret_mat);

    return real_num_div;
}

// SplitMat_B
// return real_num_div
int split_dmatrix_t_dmat(DMatrix ret_mat[], int num_div, DMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, real_total_dim;
	int real_num_div, num_digits = 53; // IEEE double prec.
	//int flag_stop = 0; //, num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//int flag_stop = 0, num_digits = 64; // IEEE double prec.
	//double *s;
    DMatrix tmp_org_mat;
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

    tmp_org_mat = init_dmatrix(row_dim, col_dim);
    subst_dmatrix(tmp_org_mat, org_mat);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        if(ret_mat != NULL)
            in_ret_mat = ret_mat[index];

        //subst_dmatrix_ddmat(ret_mat[index], tmp_org_mat);
        subst_dmatrix(in_ret_mat, tmp_org_mat);

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
        sub_dmatrix(tmp_org_mat, tmp_org_mat, in_ret_mat);
        //sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);

        real_num_div = index + 1;
    }

    // free s
	free_dmatrix(s);
    free_dmatrix(tmp_mat[0]);
    free_dmatrix(tmp_mat[1]);
    free_dmatrix(tmp_org_mat);
    if(ret_mat == NULL)
        free_dmatrix(in_ret_mat);

    return real_num_div;
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
