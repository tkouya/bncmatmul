/********************************************************************************/
/* test_mpf_oz_scheme: Multiple precision linear computation based on Ozaki scheme.  */
/* Copyright (C) 2024 Tomonori Kouya                                            */
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
#include <stdio.h>
#include <float.h> // DBL_MAX, MIN 
#include <fenv.h> // FE_NEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO
#include "matmul_strassen.h"
#include "get_secv.h"

// print MPFR rounding mode
void print_mpfr_rounding_mode(mpfr_rnd_t rnd_mode)
{
    switch(rnd_mode)
    {
        case MPFR_RNDN: printf("MPFR_RNDN: Round to nearest\n"); return;
        case MPFR_RNDZ: printf("MPFR_RNDZ: Round to zero\n"); return;
        case MPFR_RNDU: printf("MPFR_RNDU: Round to +inf\n"); return;
        case MPFR_RNDD: printf("MPFR_RNDD: Round to -inf\n"); return;
        case MPFR_RNDA: printf("MPFR_RNDA: Round away from zero\n"); return;
        case MPFR_RNDF: printf("MPFR_RNDF: Faithful rounding\n"); return;
    }
}

// split vector
// ret_vec[0] * 2^ret_vec_exp[0] + ... ret_vec[num_div - 1] * 2^ret_vec_exp[num_div - 1] = org_vec
int split_mpfvector_dex(MPFVector ret_vec[], mpfr_exp_t ret_vec_exp[], int max_num_div_ex, MPFVector org_vec)
{
    unsigned int prec = org_vec->prec;
    long int dim = org_vec->dim;
    int index, max_index, real_num_div, num_bits = 53; // IEEE754 binary64
    long int i;
    mpf_t tmp, tmp_abs, tmp_mantissa, max_abs, min_abs, max_abs_threshold, min_abs_threshold;
    mpfr_exp_t tmp_exp, max_abs_exp, min_abs_exp, width_exp;
    mpfr_rnd_t current_rnd = mpfr_get_default_rounding_mode();
    mpfr_prec_t current_prec = mpfr_get_default_prec();
    MPFVector abs_org_vec;

    // init
    mpf_init2(tmp, prec);
    mpf_init2(tmp_abs, prec);
    mpf_init2(tmp_mantissa, prec);
    mpf_init2(max_abs, prec);
    mpf_init2(min_abs, prec);
    mpf_init2(max_abs_threshold, prec);
    mpf_init2(min_abs_threshold, prec);

    // tmp_org_vec := org_vec
    abs_org_vec = init2_mpfvector(dim, prec);

    // info. of binar64(double prec.)
/*    printf("FLT_RADIX = %d\n", FLT_RADIX);
    //printf("FLT_ROUND = %d\n", FLT_ROUND);
    printf("DBL_MANT_DIG = %d\n", DBL_MANT_DIG);
    printf("DBL_DIG = %d\n", DBL_DIG);
    printf("DBL_MAX = %g\n", DBL_MAX);
    printf("DBL_MIN = %g\n", DBL_MIN);
    printf("DBL_MAX_EXP = %d\n", DBL_MAX_EXP);    
    printf("DBL_MIN_EXP = %d\n", DBL_MIN_EXP);    
    printf("DBL_EPSILON = %g\n", DBL_EPSILON);

    // current info. of MPFR
    printf("mpfr_get_version = %s\n", mpfr_get_version());
    printf("mpfr_get_default_rounding_mode: "); print_mpfr_rounding_mode(current_rnd);
    printf("mpfr_get_default_prec = %d\n", current_prec);
    printf("mpfr_get_emax = %d\n", (long int)mpfr_get_emax());
    printf("mpfr_get_emin = %d\n", (long int)mpfr_get_emin());
*/
    // check if all elements are in normal fp num
    mpfr_abs(max_abs, get_mpfvector_i(org_vec, 0), current_rnd);
    mpfr_abs(min_abs, get_mpfvector_i(org_vec, 0), current_rnd);
    for(i = 0; i < dim; i++)
    {
        // tmp_abs := abs(org_vec[i])
        mpfr_abs(tmp_abs, get_mpfvector_i(org_vec, i), current_rnd);
        set_mpfvector_i(abs_org_vec, i, tmp_abs);

        if(mpfr_cmp(tmp_abs, max_abs) > 0) mpfr_set(max_abs, tmp_abs, current_rnd);
        if(mpfr_cmp(tmp_abs, min_abs) < 0) mpfr_set(min_abs, tmp_abs, current_rnd);

        //mpfr_frexp(&tmp_exp, tmp_mantissa, get_mpfvector_i(org_vec, i), current_rnd);
        //mpfr_printf("%5d: %RNe -> %RNe * 2^%d\n", i, get_mpfvector_i(org_vec, i), tmp_mantissa, tmp_exp);
        //mpfr_printf("%5d: %RNb -> %RNb * 2^%d\n", i, get_mpfvector_i(org_vec, i), tmp_mantissa, tmp_exp);        
    }
    mpfr_frexp(&max_abs_exp, tmp_mantissa, max_abs, current_rnd);
    mpfr_frexp(&min_abs_exp, tmp_mantissa, min_abs, current_rnd);

    //mpfr_printf("max_abs_vec, exp = %RNe, %d\n", max_abs, max_abs_exp);
    //mpfr_printf("min_abs_vec, exp = %RNe, %d\n", min_abs, min_abs_exp);

    // split mpfvector according to exp
    real_num_div = (mpfr_exp_t)((max_abs_exp - min_abs_exp) / DBL_MAX_EXP) + 1;
    width_exp = (mpfr_exp_t)((max_abs_exp - min_abs_exp) / real_num_div);
    max_index = (real_num_div < max_num_div_ex) ? real_num_div : max_num_div_ex;

    //printf("real_num_div, max_num_div_ex, max_index = %d, %d, %d\n", real_num_div, max_num_div_ex, max_index);
    for(index = 0; index < max_index; index++)
    {
        set0_mpfvector(ret_vec[index]);

        // min_abs_threshold := max_abs / 2^(dbl_max_exp * real_num_div)
        // max_abs_threshold := max_abs / 2^(dbl_max_exp * (real_num_div + 1))
        ret_vec_exp[index] = max_abs_exp - DBL_MAX_EXP * index;
        mpfr_div_2ui(max_abs_threshold, max_abs, DBL_MAX_EXP * index, current_rnd);
        mpfr_div_2ui(min_abs_threshold, max_abs, DBL_MAX_EXP * (index + 1), current_rnd);
        //mpfr_printf("%d %ld [%RNe, %RNe]\n", index, ret_vec_exp[index], min_abs_threshold, max_abs_threshold);
        for(i = 0; i < dim; i++)
        {
            if(
                (mpfr_cmp(get_mpfvector_i(abs_org_vec, i), min_abs_threshold) > 0)
                && (mpfr_cmp(get_mpfvector_i(abs_org_vec, i), max_abs_threshold) <= 0)
            )
            {
                mpfr_div_2si(tmp, get_mpfvector_i(org_vec, i), ret_vec_exp[index], current_rnd);
                set_mpfvector_i(ret_vec[index], i, tmp);
                //mpfr_printf("%d, %RNe\n", i, gmpfvi(org_vec, i));
            }
        }
    }

    // free
    free_mpfvector(abs_org_vec);
    mpf_clear(tmp);
    mpf_clear(tmp_abs);
    mpf_clear(tmp_mantissa);
    mpf_clear(max_abs);
    mpf_clear(min_abs);
    mpf_clear(max_abs_threshold);
    mpf_clear(min_abs_threshold);

    return max_index;
}

// add_mpfvector_dex
void add_mpfvector_dex(MPFVector ret, MPFVector src_vec_array[], mpfr_exp_t src_vec_exp_array[], int max_num_div_ex)
{
    int index, i;
    mpf_t coef, tmp;
    mpfr_rnd_t current_rnd = mpfr_get_default_rounding_mode();
    MPFVector vec;

    mpf_init2(coef, ret->prec);
    mpf_init2(tmp, ret->prec);
    vec = init2_mpfvector(ret->dim, ret->prec);

    set0_mpfvector(ret);
    for(index = 0; index < max_num_div_ex; index++)
    {
        for(i = 0; i < src_vec_array[index]->dim; i++)
        {
            mpfr_mul_2si(tmp, get_mpfvector_i(src_vec_array[index], i), src_vec_exp_array[index], current_rnd);
            set_mpfvector_i(vec, i, tmp);
        }
        add_mpfvector(ret, ret, vec);
    }

    mpf_clear(coef);
    mpf_clear(tmp);
    free_mpfvector(vec);
}

// split matrix
// ret_mat[0] * 2^ret_mat_exp[0] + ... ret_mat[num_div - 1] * 2^ret_mat_exp[num_div - 1] = org_mat
int split_mpfmatrix_dex(MPFMatrix ret_mat[], mpfr_exp_t ret_mat_exp[], int max_num_div_ex, MPFMatrix org_mat)
{
    unsigned int prec = org_mat->prec;
    long int row_dim = org_mat->row_dim, col_dim = org_mat->col_dim;
    int index, max_index, real_num_div, num_bits = 53; // IEEE754 binary64
    long int i, j;
    mpf_t tmp, tmp_abs, tmp_mantissa, max_abs, min_abs, max_abs_threshold, min_abs_threshold;
    mpfr_exp_t tmp_exp, max_abs_exp, min_abs_exp, width_exp;
    mpfr_rnd_t current_rnd = mpfr_get_default_rounding_mode();
    mpfr_prec_t current_prec = mpfr_get_default_prec();
    MPFMatrix abs_org_mat;

    // init
    mpf_init2(tmp, prec);
    mpf_init2(tmp_abs, prec);
    mpf_init2(tmp_mantissa, prec);
    mpf_init2(max_abs, prec);
    mpf_init2(min_abs, prec);
    mpf_init2(max_abs_threshold, prec);
    mpf_init2(min_abs_threshold, prec);

    // tmp_org_vec := org_vec
    abs_org_mat = init2_mpfmatrix(row_dim, col_dim, prec);

    // check if all elements are in normal fp num
    mpfr_abs(max_abs, get_mpfmatrix_ij(org_mat, 0, 0), current_rnd);
    mpfr_abs(min_abs, get_mpfmatrix_ij(org_mat, 0, 0), current_rnd);
    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            // tmp_abs := abs(org_mat[i])
            mpfr_abs(tmp_abs, get_mpfmatrix_ij(org_mat, i, j), current_rnd);
            set_mpfmatrix_ij(abs_org_mat, i, j, tmp_abs);

            if(mpfr_cmp(tmp_abs, max_abs) > 0) mpfr_set(max_abs, tmp_abs, current_rnd);
            if(mpfr_cmp(tmp_abs, min_abs) < 0) mpfr_set(min_abs, tmp_abs, current_rnd);

        }
    }
    mpfr_frexp(&max_abs_exp, tmp_mantissa, max_abs, current_rnd);
    mpfr_frexp(&min_abs_exp, tmp_mantissa, min_abs, current_rnd);

    //mpfr_printf("max_abs_vec, exp = %RNe, %d\n", max_abs, max_abs_exp);
    //mpfr_printf("min_abs_vec, exp = %RNe, %d\n", min_abs, min_abs_exp);

    // split mpfvector according to exp
    //width_exp = (mpfr_exp_t)((max_abs_exp - min_abs_exp) / real_num_div);
    //width_exp = 53; //DBL_MANT_DIG; //DBL_MAX_EXP; // - (mpfr_exp_t)ceil(mylog2(row_dim));
    width_exp = DBL_MAX_EXP; // - (mpfr_exp_t)ceil(mylog2(row_dim));
    //real_num_div = (mpfr_exp_t)((max_abs_exp - min_abs_exp) / DBL_MAX_EXP) + 1;
    real_num_div = (mpfr_exp_t)((max_abs_exp - min_abs_exp) / width_exp) + 1;
    max_index = (real_num_div < max_num_div_ex) ? real_num_div : max_num_div_ex;

    //printf("real_num_div, max_num_div_ex, max_index = %d, %d, %d\n", real_num_div, max_num_div_ex, max_index);
    for(index = 0; index < max_index; index++)
    {
        set0_mpfmatrix(ret_mat[index]);

        // min_abs_threshold := max_abs / 2^(dbl_max_exp * real_num_div)
        // max_abs_threshold := max_abs / 2^(dbl_max_exp * (real_num_div + 1))
        /*
        ret_mat_exp[index] = max_abs_exp - DBL_MAX_EXP * index;
        mpfr_div_2ui(max_abs_threshold, max_abs, DBL_MAX_EXP * index, current_rnd);
        mpfr_div_2ui(min_abs_threshold, max_abs, DBL_MAX_EXP * (index + 1), current_rnd);
        */
        ret_mat_exp[index] = max_abs_exp - width_exp * index;
        mpfr_div_2ui(max_abs_threshold, max_abs, width_exp * index, current_rnd);
        mpfr_div_2ui(min_abs_threshold, max_abs, width_exp * (index + 1), current_rnd);
        //mpfr_printf("%d %ld [%RNe, %RNe]\n", index, ret_mat_exp[index], min_abs_threshold, max_abs_threshold);
        for(i = 0; i < row_dim; i++)
        {
            for(j = 0; j < col_dim; j++)
            {
                if(
                    (mpfr_cmp(get_mpfmatrix_ij(abs_org_mat, i, j), min_abs_threshold) > 0)
                    && (mpfr_cmp(get_mpfmatrix_ij(abs_org_mat, i, j), max_abs_threshold) <= 0)
                )
                {
                    mpfr_div_2si(tmp, get_mpfmatrix_ij(org_mat, i, j), ret_mat_exp[index], current_rnd);
                    set_mpfmatrix_ij(ret_mat[index], i, j, tmp);
                    //mpfr_printf("%d: %d, %d, %25.17RNe -> %25.17RNe\n", index, i, j, gmpfmij(org_mat, i, j), tmp);
                }
            }
        }
    }

    // free
    free_mpfmatrix(abs_org_mat);
    mpf_clear(tmp);
    mpf_clear(tmp_abs);
    mpf_clear(tmp_mantissa);
    mpf_clear(max_abs);
    mpf_clear(min_abs);
    mpf_clear(max_abs_threshold);
    mpf_clear(min_abs_threshold);

    return max_index;
}

// add_mpfvector_dex
void add_mpfmatrix_dex(MPFMatrix ret, MPFMatrix src_mat_array[], mpfr_exp_t src_mat_exp_array[], int max_num_div_ex)
{
    int index, i, j;
    mpf_t coef, tmp;
    mpfr_rnd_t current_rnd = mpfr_get_default_rounding_mode();
    MPFMatrix mat;

    mpf_init2(coef, ret->prec);
    mpf_init2(tmp, ret->prec);
    mat = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);

    set0_mpfmatrix(ret);
    for(index = 0; index < max_num_div_ex; index++)
    {
        for(i = 0; i < src_mat_array[index]->row_dim; i++)
        {
            for(j = 0; j < src_mat_array[index]->col_dim; j++)
            {
                mpfr_mul_2si(tmp, get_mpfmatrix_ij(src_mat_array[index], i, j), src_mat_exp_array[index], current_rnd);
                set_mpfmatrix_ij(mat, i, j, tmp);
            }
        }
        add_mpfmatrix(ret, ret, mat);
    }

    mpf_clear(coef);
    mpf_clear(tmp);
    free_mpfmatrix(mat);
}

// mul_mpfmatrix_oz_dex
void mul_mpfmatrix_oz_dex(MPFMatrix ret, MPFMatrix mat_a, int max_num_div_a, MPFMatrix mat_b,  int max_num_div_b, int max_num_div_ex_a, int max_num_div_ex_b)
{
    int i, j, in_i, in_j;
    int real_div_num_mat_a, real_div_num_mat_b;
    MPFMatrix *mat_a_array, *mat_b_array;
    MPFMatrix tmp_mat, tmp_mat_oz;
    mpf_t coef, tmp, relerr[3];
    mpfr_exp_t *mat_a_array_exp, *mat_b_array_exp, ab_exp;
    mpfr_rnd_t current_rnd = mpfr_get_default_rounding_mode();

    // Initialize
    mpf_init2(coef, ret->prec);
    mpf_init2(tmp, ret->prec);
    mpf_init2(relerr[0], ret->prec); mpf_init2(relerr[1], ret->prec); mpf_init2(relerr[2], ret->prec);
    tmp_mat = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    tmp_mat_oz = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    mat_a_array = (MPFMatrix *)calloc((size_t)max_num_div_ex_a, sizeof(MPFMatrix));
    mat_b_array = (MPFMatrix *)calloc((size_t)max_num_div_ex_b, sizeof(MPFMatrix));
    mat_a_array_exp = (mpfr_exp_t *)calloc((size_t)max_num_div_ex_a, sizeof(mpfr_exp_t));
    mat_b_array_exp = (mpfr_exp_t *)calloc((size_t)max_num_div_ex_b, sizeof(mpfr_exp_t));
    for(i = 0; i < max_num_div_ex_a; i++)
    {
        mat_a_array[i] = init2_mpfmatrix(mat_a->row_dim, mat_a->col_dim, mat_a->prec);
        mat_a_array_exp[i] = 0;
    }
    for(i = 0; i < max_num_div_ex_b; i++)
    {
        mat_b_array[i] = init2_mpfmatrix(mat_b->row_dim, mat_b->col_dim, mat_b->prec);
        mat_b_array_exp[i] = 0;
    }

    // Split matrices to them within binary64
    real_div_num_mat_a = split_mpfmatrix_dex(mat_a_array, mat_a_array_exp, max_num_div_ex_a, mat_a);
    real_div_num_mat_b = split_mpfmatrix_dex(mat_b_array, mat_b_array_exp, max_num_div_ex_b, mat_b);

    // C := \SUM_i \SUM_j (2^exp_a_i * A_i) * (2^exp_b_j) * B_j
    set0_mpfmatrix(ret);
    for(i = 0; i < real_div_num_mat_a; i++)
    {
        for(j = 0; j < real_div_num_mat_b; j++)
        {
            //mpf_set_ui(coef, 1UL);
            ab_exp =  mat_a_array_exp[i] + mat_b_array_exp[j];

            mul_mpfmatrix_oz(tmp_mat_oz, mat_a_array[i], max_num_div_a, mat_b_array[j], max_num_div_b);
            //mul_mpfmatrix_oz(tmp_mat, mat_a_array[i], max_num_div_a, mat_b_array[j], max_num_div_b);
            //mul_mpfmatrix(tmp_mat_oz, mat_a_array[i], mat_b_array[j]);
            mul_mpfmatrix(tmp_mat, mat_a_array[i], mat_b_array[j]);
            relerr3_mpfmatrix(relerr[0], relerr[1], relerr[2], tmp_mat_oz, tmp_mat, 2);
            //mpfr_printf("%d, %d: %10.3RNe, %10.3RNe, %10.3RNe\n", i, j, relerr[0], relerr[1], relerr[2]);
            //cmul_mpfmatrix(tmp_mat, coef, tmp_mat);
            for(in_i = 0; in_i < tmp_mat->row_dim; in_i++)
            {
                for(in_j = 0; in_j < tmp_mat->col_dim; in_j++)
                {
                    mpfr_mul_2si(tmp, get_mpfmatrix_ij(tmp_mat, in_i, in_j), ab_exp, current_rnd);
                    set_mpfmatrix_ij(tmp_mat, in_i, in_j, tmp);
                }
            }

            add_mpfmatrix(ret, ret, tmp_mat);
            //print_mpfmatrix(tmp_mat);
        }
    }

    // Free
    mpf_clear(coef);
    mpf_clear(tmp);
    mpf_clear(relerr[0]); mpf_clear(relerr[1]); mpf_clear(relerr[2]);
    free_mpfmatrix(tmp_mat);
    free_mpfmatrix(tmp_mat_oz);
    for(i = 0; i < max_num_div_ex_a; i++)
        free_mpfmatrix(mat_a_array[i]);
    for(i = 0; i < max_num_div_ex_b; i++)
        free_mpfmatrix(mat_b_array[i]);

    free(mat_a_array);
    free(mat_b_array);
    free(mat_a_array_exp);
    free(mat_b_array_exp);

}

//#define dim 5
//#define DIM 10
#define DIM 128
//#define DIM 512

//#define MAX_NUM_DIV 10
//#define MAX_NUM_DIV 15
//#define MAX_NUM_DIV 128
#define MAX_NUM_DIV 8192

int main(int argc, char *argv[])
{
    unsigned long prec;
    long int i, j, dim;
    int real_num_div, num_div, real_num_exp_div;
    mpf_t tmp, tmp2;
    MPFVector vec_org, vec, vec_array[MAX_NUM_DIV];
    DVector dvec[MAX_NUM_DIV];
    MPFMatrix mat_a, mat_a_array[MAX_NUM_DIV], mat_b, mat_b_array[MAX_NUM_DIV], mat_c, mat;
    MPFMatrix mat_a_long, mat_b_long, mat_c_long;
    DMatrix dmat[MAX_NUM_DIV];
    mpf_t ddtmp, ddtmp2, relerr[3];
    mpfr_exp_t vec_array_exp[MAX_NUM_DIV], mat_a_array_exp[MAX_NUM_DIV], mat_b_array_exp[MAX_NUM_DIV];
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
    goto mpfmatrix;

// DVector 
    vec_org = init_mpfvector(dim);
    vec = init_mpfvector(dim);
    for(i = 0; i < num_div; i++)
    {
        dvec[i] = init_dvector(dim);
        vec_array[i] = init_mpfvector(dim);
    }

    // set random
    srand(dim);
    for(i = 0; i < dim; i++)
    {
        mpf_set_d(tmp, (double)rand());
        mpf_mul_d(tmp, tmp, (double)rand());
        mpf_div_d(tmp, tmp, (double)rand());
        if(rand() % 2 == 1) mpf_neg(tmp, tmp);
        if(rand() % 2 == 1)
            mpfr_mul_2si(tmp, tmp, rand() % 65536, MPFR_RNDN);
        else 
            mpfr_div_2si(tmp, tmp, rand() % 65536, MPFR_RNDN);
        
        set_mpfvector_i(vec_org, i, tmp);
    }

    // split
    real_num_exp_div = split_mpfvector_dex(vec_array, vec_array_exp, num_div, vec_org);
    //real_num_div = split_mpfvector_dvec(dvec, num_div, vec_org); //, 53);
    printf("real_num_exp_div = %d\n", real_num_exp_div);
    for(i = 0; i < real_num_exp_div; i++)
    {
        real_num_div = split_mpfvector_dvec(dvec, num_div, vec_array[i]); //, 53);
        set0_mpfvector(vec);
        for(j = 0; j < real_num_div; j++)
        {
            //printf("dvec[%d]:\n", i); print_dvector(dvec[i]);

            add_mpfvector_dvec(vec, vec, dvec[j]);
        }
        printf("vec_array_exp[%d] = %d\n", i, vec_array_exp[i]);
        printf("vec_array[%d] = \n", i);
        print_mpfvector(vec);
    }

    add_mpfvector_dex(vec, vec_array, vec_array_exp, real_num_exp_div);
    printf("vec = \n");
    print_mpfvector(vec); 
    sub_mpfvector(vec, vec, vec_org);
    printf("vec = \n");
    print_mpfvector(vec);
    norm2_mpfvector(tmp, vec);
    mpfr_printf("||vec - vec_org||_2 = %10.3RNe\n", tmp);

    free_mpfvector(vec_org);
    free_mpfvector(vec);
    for(i = 0; i < 10; i++)
    {
        free_dvector(dvec[i]);
        free_mpfvector(vec_array[i]);
    }

    //sgoto end;
// DMatrix
mpfmatrix:

    mpf_init(relerr[0]);
    mpf_init(relerr[1]);
    mpf_init(relerr[2]);

    mat_a = init_mpfmatrix(dim, dim);
    mat_b = init_mpfmatrix(dim, dim);
    mat_c = init_mpfmatrix(dim, dim);
    mat_a_long = init2_mpfmatrix(dim, dim, (unsigned long)((double)prec * 1.5));
    mat_b_long = init2_mpfmatrix(dim, dim, (unsigned long)((double)prec * 1.5));
    mat_c_long = init2_mpfmatrix(dim, dim, (unsigned long)((double)prec * 1.5));
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
            if(rand() % 2 == 1) mpf_neg(tmp, tmp);
            
            if(rand() % 2 == 1) mpfr_mul_2si(tmp, tmp, rand() % 4096, MPFR_RNDN);
            else mpfr_div_2si(tmp, tmp, rand() % 4096, MPFR_RNDN);

            set_mpfmatrix_ij(mat_a, i, j, tmp);

            mpf_set_d(tmp, (double)rand() / (double)RAND_MAX);
            mpf_mul_d(tmp, tmp, (double)rand());
            mpf_div_d(tmp, tmp, (double)rand());

            if(rand() % 2 == 1) mpfr_mul_2si(tmp, tmp, rand() % 4096, MPFR_RNDN);
            else mpfr_div_2si(tmp, tmp, rand() % 4096, MPFR_RNDN);

            set_mpfmatrix_ij(mat_b, i, j, tmp);
        }
    }

    // init mat_a_array
    for(i = 0; i < MAX_NUM_DIV; i++)
    {
        mat_a_array[i] = init_mpfmatrix(dim, dim);
        mat_b_array[i] = init_mpfmatrix(dim, dim);
    }

    // split_A
    printf("Split_A:\n");
    real_num_div = split_mpfmatrix_dex(mat_a_array, mat_a_array_exp, num_div, mat_a);
    printf("num_div, real_num_div = %d, %d\n", num_div, real_num_div);
    add_mpfmatrix_dex(mat, mat_a_array, mat_a_array_exp, real_num_div);
    relerr3_mpfmatrix(relerr[0], relerr[1], relerr[2], mat, mat_a, 2);
    mpfr_printf("A: Max_relerr, Min_relerr, NormF_relerr = %10.3RNe, %10.3RNe, %10.3RNe\n", relerr[0], relerr[1], relerr[2]);
    printf("mat_a_array: ");
    for(i = 0; i < real_num_div; i++)
        printf("%d ", split_mpfmatrix_dmat(dmat, num_div, mat_a_array[i]));
    printf("\n");

    // split_B
    printf("Split_B:\n");
    real_num_div = split_mpfmatrix_dex(mat_b_array, mat_b_array_exp, num_div, mat_b);
    printf("num_div, real_num_div = %d, %d\n", num_div, real_num_div);
    add_mpfmatrix_dex(mat, mat_b_array, mat_b_array_exp, real_num_div);
    relerr3_mpfmatrix(relerr[0], relerr[1], relerr[2], mat, mat_b, 2);
    mpfr_printf("B: Max_relerr, Min_relerr, NormF_relerr = %10.3RNe, %10.3RNe, %10.3RNe\n", relerr[0], relerr[1], relerr[2]);




    // a * b
    stime = get_secv();
    //mul_mpfmatrix_oz(mat, mat_a, num_div, mat_b, num_div);
    mul_mpfmatrix_oz_dex(mat, mat_a, num_div, mat_b, num_div, num_div, num_div);
    //mul_mpfmatrix(mat, mat_a, mat_b);
    etime = get_secv() - stime;
    //print_mpfmatrix(mat);
    printf("mul_mpfmatrix_oz(dim, prec, sec)      : %5d, %5d, %10.3g\n", dim, prec, etime);

    subst_mpfmatrix(mat_a_long, mat_a);
    subst_mpfmatrix(mat_b_long, mat_b);
    stime = get_secv();
    //mul_mpfmatrix(mat_c, mat_a, mat_b);
    mul_mpfmatrix_strassen(mat_c_long, mat_a_long, mat_b_long, 32);
    etime = get_secv() - stime;
    //print_mpfmatrix(mat_c_long);
    printf("mul_mpfmatrix_strassen(dim, prec, sec): %5d, %5d, %10.3g\n", dim, prec, etime);

    relerr3_mpfmatrix(relerr[0], relerr[1], relerr[2], mat, mat_c_long, 2);
    mpfr_printf("Max_relerr, Min_relerr, NormF_relerr = %10.3RNe, %10.3RNe, %10.3RNe\n", relerr[0], relerr[1], relerr[2]);

    //sub_mpfmatrix(mat_c_long, mat_c_long, mat); print_mpfmatrix(mat_c_long);

    printf("\n"); 

    free_mpfmatrix(mat_a);
    free_mpfmatrix(mat_b);
    free_mpfmatrix(mat_c);
    free_mpfmatrix(mat);
    for(i = 0; i < 10; i++)
        free_dmatrix(dmat[i]);

    mpf_clear(tmp); mpf_clear(tmp2);
    mpf_clear(ddtmp); mpf_clear(ddtmp2);
    mpf_clear(relerr[0]); mpf_clear(relerr[1]); mpf_clear(relerr[2]);

end:
    return 0;
}
