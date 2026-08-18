/********************************************************************************/
/* dd_oz_scheme: Multiple precision linear computation based on Ozaki scheme.   */
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
double absmax_dvector(long int *max_index, DVector vec)
{
    long int i, max_i, dim = vec->dim;
    double ret, abs_val;

    max_i = 0;
    ret = 0.0;
    for(i = 0; i < dim; i++)
    {
        abs_val = fabs(get_dvector_i(vec, i));
        if(ret < abs_val)
        {
            ret = abs_val;
            max_i = i;
        }
    }

    if(max_index != NULL)
        *max_index = max_i;

    return ret;
}

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

#ifdef USE_MPFR
void rdd_pow_mpfr(double ret[], double a[], double b[])
{
    mpfr_t in_a, in_b, in_ret;

    mpfr_init2(in_a, 53 * DDSIZE); mpfr_set_dd(in_a, a, MPFR_RNDN);
    mpfr_init2(in_b, 53 * DDSIZE); mpfr_set_dd(in_b, b, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * DDSIZE);

    mpfr_pow(in_ret, in_a, in_b, MPFR_RNDN);
    mpfr_get_dd(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_a);
    mpfr_clear(in_b);
    mpfr_clear(in_ret);
}
#endif // USE_MPFR

// extract vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
void extract_ddvector(DVector ret_vec[], int num_div, DDVector org_vec, int num_bits)
{
    long int dim = org_vec->dim;
    int index;
    long int i;
    double org_vec_i, ret_high_vec_i, tmp[DDSIZE];
    double absmax_org_vec, threshold, t_exp; 
    DDVector tmp_org_vec;

    // tmp_org_vec := org_vec
    tmp_org_vec = init_ddvector(dim);
    subst_ddvector(tmp_org_vec, org_vec);

    for(index = 0; index < num_div; index++)
    {
        subst_dvector_ddvec(ret_vec[index], tmp_org_vec);
        absmax_org_vec = absmax_dvector(NULL, ret_vec[index]);

        // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
        t_exp = ceil(DLOG2(absmax_org_vec)) + ceil(((double)num_bits + DLOG2((double)(dim + 1))) / 2.0);
        
        //rdd_pow(threshold, two, t_exp);
        //rdd_pow_mpfr(threshold, two, t_exp);
        threshold = pow(2.0, t_exp);

        for(i = 0; i < dim; i++)
        {
            // set high vector
            //rdd_set(org_vec_i, get_ddvector_i(tmp_org_vec, i)); 
            org_vec_i = get_dvector_i(ret_vec[index], i);   
            ret_high_vec_i = org_vec_i + threshold;
            //rdd_add(ret_high_vec_i, org_vec_i, threshold);
            ret_high_vec_i -= threshold;
            //rdd_sub(ret_high_vec_i, ret_high_vec_i, threshold);
            set_dvector_i(ret_vec[index], i, ret_high_vec_i);

            // set low vector
            rdd_sub_d(tmp, get_ddvector_i(tmp_org_vec, i), ret_high_vec_i);           
            set_ddvector_i(tmp_org_vec, i, tmp);
        }
    }

    free_ddvector(tmp_org_vec);
}

// absmax_row_ddmatrix
void absmax_row_ddmatrix(double mu[DDSIZE], long int *max_j, long int row_index, DDMatrix mat)
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

// absmax_row_dmatrix
double absmax_row_dmatrix(long int *max_j, long int row_index, DMatrix mat)
{
    long int j, max_index = 0;
    double mu, abs_aij;

	mu = fabs(get_dmatrix_ij(mat, row_index, 0));
    //rdd_abs(mu, get_ddmatrix_ij(mat, row_index, 0));

	for(j = 1; j < mat->col_dim; j++)
	{
		//rdd_abs(abs_aij, get_ddmatrix_ij(mat, row_index, j));
		abs_aij = fabs(get_dmatrix_ij(mat, row_index, j));
		//if(rdd_cmp(abs_aij, mu) > 0)
		if(abs_aij > mu)
        {
			mu = abs_aij;
            //rdd_set(mu, abs_aij);
            max_index = j;
        }
	}

    if(max_j != NULL)
        *max_j = max_index;

    return mu;
    //return;
}

/* c := a + (doble)b */
void add_ddmatrix_dmat(DDMatrix c, DDMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

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
			rdd_add_d(tmp, get_ddmatrix_ij(a, i, j), get_dmatrix_ij(b, i, j));
			set_ddmatrix_ij(c, i, j, tmp);
		}
	}
}

/* c := a - (doble)b */
void sub_ddmatrix_dmat(DDMatrix c, DDMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

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
			rdd_sub_d(tmp, get_ddmatrix_ij(a, i, j), get_dmatrix_ij(b, i, j));
			set_ddmatrix_ij(c, i, j, tmp);
		}
	}
}

//#define SPLIT_NUM_DIGITS 55
#define SPLIT_NUM_DIGITS 64

// SplitMat_A
void split_ddmatrix(DMatrix ret_mat[], int num_div, DDMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim;
	//long int num_digits = 53; // IEEE double prec.
	long int num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//long int num_digits = 64; // IEEE double prec.
	//double *s;
    DDMatrix tmp_org_mat;
    DMatrix s, tmp_mat[2]; 
	double mu, abs_aij, t_exp, power2;

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;

	// threshold matrix 
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    //s = init_ddmatrix(row_dim, col_dim);
    s = init_dmatrix(row_dim, col_dim);

    //tmp_mat[0] = init_ddmatrix(row_dim, col_dim);
    //tmp_mat[1] = init_ddmatrix(row_dim, col_dim);
    tmp_mat[0] = init_dmatrix(row_dim, col_dim);
    tmp_mat[1] = init_dmatrix(row_dim, col_dim);

    // tmp_org_mat := org_mat;
    tmp_org_mat = init_ddmatrix(row_dim, col_dim);
    subst_ddmatrix(tmp_org_mat, org_mat);

    for(index = 0; index < num_div; index++)
    {
        //printf("In split_ddmatrix ... index= %d\n", index);
        //set0_dmatrix(ret_mat[index]);
        subst_dmatrix_ddmat(ret_mat[index], tmp_org_mat);

        // mu[i] = max_j |mat[i, j]|
        for(i = 0; i < row_dim; i++)
        {
            //absmax_row_ddmatrix(mu, NULL, i, tmp_org_mat);
            mu = absmax_row_dmatrix(NULL, i, ret_mat[index]);

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            //t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);
            //t_exp[1] = 0.0;
            t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);

            // s[i, j] = 2^t_exp
            //rdd_pow(power2, two, t_exp);
            //rdd_pow_mpfr(power2, two, t_exp);
            power2 = pow(2.0, t_exp);
            //printf("index, i, power2 = %ld, %ld, %25.17e\n", index, i, power2[0]);
            for(j = 0; j < col_dim; j++)
            {
                //s[i * col_dim + j] = pow(2.0, t_exp);
                //set_ddmatrix_ij(s, i, j, power2);
                set_dmatrix_ij(s, i, j, power2);
            }
        }

        // split org_mat to ret_high_mat and ret_low_mat

        // tmp_mat := mat + s
        //add_ddmatrix(tmp_mat[0], tmp_org_mat, s);
        add_dmatrix(tmp_mat[0], ret_mat[index], s);

        // high_mat := tmp_mat - s
        //sub_ddmatrix(tmp_mat[1], tmp_mat[0], s);
        sub_dmatrix(tmp_mat[1], tmp_mat[0], s);
        subst_dmatrix(ret_mat[index], tmp_mat[1]);


        // low_mat := mat - high_mat
        //sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
        sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);
    }

	// free s
	//free_ddmatrix(s);
	free_dmatrix(s);
    free_ddmatrix(tmp_org_mat);
    free_dmatrix(tmp_mat[0]);
    free_dmatrix(tmp_mat[1]);
}

// absmax_col_ddmatrix
void absmax_col_ddmatrix(double mu[DDSIZE], long int *max_i, long int col_index, DDMatrix mat)
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

// absmax_col_dmatrix
double absmax_col_dmatrix(long int *max_i, long int col_index, DMatrix mat)
{
    long int i, max_index = 0;
    double mu, abs_aij;

	mu = fabs(get_dmatrix_ij(mat, 0, col_index));
    //rdd_abs(mu, get_ddmatrix_ij(mat, 0, col_index));
	for(i = 1; i < mat->row_dim; i++)
	{
		//rdd_abs(abs_aij, get_ddmatrix_ij(mat, i, col_index));
        abs_aij = fabs(get_dmatrix_ij(mat, i, col_index));
		//if(rdd_cmp(abs_aij, mu) > 0)
		if(abs_aij > mu)
        {
			//rdd_set(mu, abs_aij);
            abs_aij = mu;
            max_index = i;
        }
	}

    if(max_i != NULL)
        *max_i = max_index;

    //return;
    return mu;
}


// SplitMat_B
void split_ddmatrix_t(DMatrix ret_mat[], int num_div, DDMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim;
	//int flag_stop = 0, num_digits = 53; // IEEE double prec.
	int flag_stop = 0, num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//int flag_stop = 0, num_digits = 64; // IEEE double prec.
	//double *s;
    DDMatrix tmp_org_mat;
    DMatrix s, tmp_mat[2];
	//double mu[DDSIZE], abs_aij[DDSIZE], t_exp[DDSIZE], power2[DDSIZE], two[DDSIZE] = {2.0, 0.0};
	double mu, abs_aij, t_exp, power2;

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;

	// initialize s
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    //s = init_ddmatrix(row_dim, col_dim);
    s = init_dmatrix(row_dim, col_dim);

    tmp_mat[0] = init_dmatrix(row_dim, col_dim);
    tmp_mat[1] = init_dmatrix(row_dim, col_dim);

    tmp_org_mat = init_ddmatrix(row_dim, col_dim);
    subst_ddmatrix(tmp_org_mat, org_mat);

    for(index = 0; index < num_div; index++)
    {
        subst_dmatrix_ddmat(ret_mat[index], tmp_org_mat);
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
            mu = absmax_col_dmatrix(NULL, j, ret_mat[index]);
            //printf("mu%d: %15.7e ", j, mu);

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            //t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);
            //t_exp[1] = 0.0;
            t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);
            if(isnan(t_exp))
                flag_stop = 1;

            // s[i, j] = 2^t_exp
            //rdd_pow(power2, two, t_exp);
            //rdd_pow_mpfr(power2, two, t_exp);
            power2 = pow(2.0, t_exp);

            //printf("index, j, power2 = %ld, %ld, %25.17e\n", index, j, power2[0]);
            for(i = 0; i < row_dim; i++)
                set_dmatrix_ij(s, i, j, power2);
                //s[i * col_dim + j] = pow(2.0, t_exp);
        }
        if(flag_stop == 1)
            break;

        // tmp_mat := mat + s
        add_dmatrix(tmp_mat[0], ret_mat[index], s);

        // high_mat := tmp_mat - s
        sub_dmatrix(tmp_mat[1], tmp_mat[0], s);
        subst_dmatrix(ret_mat[index], tmp_mat[1]);

        // low_mat := mat - high_mat
        sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);
    }

    // free s
	free_dmatrix(s);
    free_dmatrix(tmp_mat[0]);
    free_dmatrix(tmp_mat[1]);
    free_ddmatrix(tmp_org_mat);
}

// Matrix multiplication based on Ozaki scheme
void mul_ddmatrix_oz(DDMatrix ret, DDMatrix a, DDMatrix b, int num_div)
{
    int i, j;
    long int row_dim = ret->row_dim, col_dim = ret->col_dim;
    DMatrix *div_a, *div_b, div_ret;

    div_a = (DMatrix *)calloc(num_div, sizeof(DMatrix));
    div_b = (DMatrix *)calloc(num_div, sizeof(DMatrix));
    for(i = 0; i < num_div; i++)
    {
        div_a[i] = init_dmatrix(row_dim, col_dim);
        div_b[i] = init_dmatrix(row_dim, col_dim);
    }
    div_ret = init_dmatrix(row_dim, col_dim);

    split_ddmatrix(div_a, num_div, a);
    split_ddmatrix_t(div_b, num_div, b);

    set0_ddmatrix(ret);
    for(i = 0; i < num_div; i++)
    {
        for(j = 0; j < num_div; j++)
        {
            mul_dmatrix(div_ret, div_a[i], div_b[j]);
            add_ddmatrix_dmat(ret, ret, div_ret);
       }
    }

    free_dmatrix(div_ret);
    for(i = 0; i < num_div; i++)
    {
        free_dmatrix(div_a[i]);
        free_dmatrix(div_b[i]);
    }
    free(div_a);
    free(div_b);

}

//#define DIM 5
//#define DIM 10
//#define DIM 100
#define DIM 512
//#define DIM 1024


//#define MAX_NUM_DIV 2
//#define MAX_NUM_DIV 5
#define MAX_NUM_DIV 6
//#define MAX_NUM_DIV 10
//#define MAX_NUM_DIV 20

int main()
{
    long int i, j;
    double tmp[DDSIZE], tmp2[DDSIZE];
    DDVector vec_org, vec;
    DVector dvec[MAX_NUM_DIV];
    DDMatrix mat_a, mat_b, mat_c, mat;
    DMatrix dmat[MAX_NUM_DIV];
    ddfloat ddtmp, ddtmp2;

    //goto ddmatrix;

// DVector 
    vec_org = init_ddvector(DIM);
    vec = init_ddvector(DIM);
    for(i = 0; i < MAX_NUM_DIV; i++)
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
        //printf("dvec[%d]:\n", i); print_dvector(dvec[i]);

        add_ddvector_dvec(vec, vec, dvec[i]);
    }
    //sub_ddvector(vec, vec, vec_org);
    //printf("vec - vec_org:\n"); print_ddvector(vec);
    //printf("\n");
    for(i = 0; i < DIM; i++)
    {
        ddtmp = get_ddvector_i_ddfloat(vec, i);
        ddtmp2 = get_ddvector_i_ddfloat(vec_org, i);
        printf("%5d %25.17e %25.17e\n%5d %25.17e %25.17e\n", i, ddtmp.val[0], ddtmp.val[1], i, ddtmp2.val[0], ddtmp2.val[1]);
    }
    

    free_ddvector(vec_org);
    free_ddvector(vec);
    for(i = 0; i < MAX_NUM_DIV; i++)
        free_dvector(dvec[i]);

    //sgoto end;
// DMatrix
ddmatrix:

    mat_a = init_ddmatrix(DIM, DIM);
    mat_b = init_ddmatrix(DIM, DIM);
    mat_c = init_ddmatrix(DIM, DIM);
    mat = init_ddmatrix(DIM, DIM);
    for(i = 0; i < MAX_NUM_DIV; i++)
        dmat[i] = init_dmatrix(DIM, DIM);

    // set random
    srand(DIM);
    for(i = 0; i < DIM; i++)
    {
        for(j = 0; j < DIM; j++)
        {
            rdd_set_d(tmp, (double)rand() / (double)RAND_MAX);
            rdd_mul_d(tmp, tmp, (double)rand());
            set_ddmatrix_ij(mat_a, i, j, tmp);

            rdd_set_d(tmp, (double)rand() / (double)RAND_MAX);
            rdd_mul_d(tmp, tmp, (double)rand());
            set_ddmatrix_ij(mat_b, i, j, tmp);
        }
    }

    // split_A
    printf("Split_A:\n");
    split_ddmatrix(dmat, MAX_NUM_DIV, mat_a);

    for(i = 0; i < MAX_NUM_DIV; i++)
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
    printf("Split_B:\n");
    split_ddmatrix_t(dmat, MAX_NUM_DIV, mat_b);

    for(i = 0; i < MAX_NUM_DIV; i++)
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
    mul_ddmatrix_oz(mat, mat_a, mat_b, MAX_NUM_DIV);
    mul_ddmatrix(mat_c, mat_a, mat_b);

    sub_ddmatrix(mat, mat, mat_c);
    printf("||mat_oz - mat_org|| / ||mat_org||:\n"); 
    normf_ddmatrix(tmp, mat);
    normf_ddmatrix(tmp2, mat_c);
    rdd_div(tmp, tmp, tmp2);
    rdd_out_str(tmp); //print_ddmatrix(mat); 
    printf("\n"); 

    free_ddmatrix(mat_a);
    free_ddmatrix(mat_b);
    free_ddmatrix(mat_c);
    free_ddmatrix(mat);
    for(i = 0; i < MAX_NUM_DIV; i++)
        free_dmatrix(dmat[i]);

end:
    return 0;
}