/********************************************************************************/
/* ctdlinear.c: Triple-double precision Complex Linear Computation Library      */
/* Copyright (C) 2023 Tomonori Kouya                                            */
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
#include "ctdlinear.h"
//#include "clinear.h"
#include "oz_scheme.h"

#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dtq_sd.h"
#include "mpflinear.h"
#endif //USE_GMP//

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// initialize CTDVector
CTDVector init_ctdvector(int dimension)
{
    CTDVector ret = NULL;

    ret = (CTDVector)malloc(sizeof(ctdvector));

    if(ret == NULL)
    {
        fprintf(stderr, "ERROR: init_ctdvector(%d)\n", dimension);
        return NULL;
    }

    ret->re = init_tdvector(dimension);
    if(ret->re == NULL)
    {
        fprintf(stderr, "ERROR: init_ctdvector(%d)\n", dimension);
        free(ret);
        return NULL;
    }

    ret->im = init_tdvector(dimension);
    if(ret->im == NULL)
    {
        fprintf(stderr, "ERROR: init_ctdvector(%d)\n", dimension);
        free_tdvector(ret->re);
        free(ret);
        return NULL;
    }

    return ret;
}

// free CTDVector
void free_ctdvector(CTDVector vec)
{
    free_tdvector(vec->re);
    free_tdvector(vec->im);
    free(vec);
}

// CTDVector vec -> tdfloat array
void set_ctdfloat_ctdvec(ctdfloat ret[], int ret_dim, CTDVector vec)
{
    int i;
    ctdfloat ptr_val;

    for(i = 0; i < ret_dim; i++)
    {
        subst_ctdvector_i(&ptr_val, vec, i);
        rctd_set(&ret[i], &ptr_val); // get_ctdvector_i(vec, i));
    }

/*    {
        ptr_val = get_ctdvector_i_ctdfloat(vec, i);
        rctd_set(&ret[i], &ptr_val);
    }
*/

    return;
}

// ctdfloat array -> CTDVector ret
void set_ctdvector_ctdfloat(CTDVector ret, ctdfloat array[], int array_dim)
{
    int i;

    for(i = 0; i < ret->re->dim; i++)
        set_ctdvector_i(ret, i, &array[i]);
}

// tdvector -> ctdvector
void set_ctdvector_tdvec(CTDVector ret, TDVector re_vec, TDVector im_vec)
{
    subst_tdvector(ret->re, re_vec);
    subst_tdvector(ret->im, im_vec);
}

// print tdvector
void print_ctdvector(CTDVector vec)
{
    int i;

    for(i = 0; i < vec->re->dim; i++)
    {
        printf("%5d ", i);
        rtd_out_str(get_tdvector_i(vec->re, i));
        printf(" + ");
        rtd_out_str(get_tdvector_i(vec->im, i));
        printf("\n");
    }
}

// set a zero vector
void set0_ctdvector(CTDVector vec)
{
    set0_tdvector(vec->re);
    set0_tdvector(vec->im);
}

/*************************************************/
/* Vector Calculations for CTDVector               */
/*
void add_ctdvector(CTDVector c, CTDVector a, CTDVector b)
void add2_ctdvector(CTDVector c, CTDVector a)
void sub_ctdvector(CTDVector c, CTDVector a, CTDVector b)
void sub2_ctdvector(CTDVector c, DVector a)
void cmul_ctdvector(CTDVector c, double val[TDSIZE], CTDVector a)
void cmul2_ctdvector(CTDVector c, double val[TDSIZE])
void add_cmul_ctdvector(CTDVector c, CTDVector a, double val[TDSIZE], CTDVector b)
double ip_ctdvector(CTDVector a, CTDVector b)
double norm1_ctdvector(CTDVector a)
double norm2_ctdvector(CTDVector a)
double normi_ctdvector(CTDVector a)
void subst_ctdvector(CTDVector c, CTDVector a)
*/
/*************************************************/
/* c = a + b */
void add_ctdvector(CTDVector c, CTDVector a, CTDVector b)
{
    add_tdvector(c->re, a->re, b->re);
    add_tdvector(c->im, a->im, b->im);
}

/* c += a */
void add2_ctdvector(CTDVector c, CTDVector a)
{
    add2_tdvector(c->re, a->re);
    add2_tdvector(c->im, a->im);
}

/* c = a - b */
void sub_ctdvector(CTDVector c, CTDVector a, CTDVector b)
{
    sub_tdvector(c->re, a->re, b->re);
    sub_tdvector(c->im, a->im, b->im);
}

/* c -= a */
void sub2_ctdvector(CTDVector c, CTDVector a)
{
    sub2_tdvector(c->re, a->re);
    sub2_tdvector(c->im, a->im);
}

/* c = val * a */
void cmul_ctdvector_4m(CTDVector c, ctdfloat *val, CTDVector a)
{
    TDVector t1, t2, t3, t4;

    // 2024-11-28(Thu) Fixed! T.Kouya
    t1 = init_tdvector(c->re->dim);
    t2 = init_tdvector(c->re->dim);
    t3 = init_tdvector(c->re->dim);
    t4 = init_tdvector(c->re->dim);

    cmul_tdvector(t1, val->val_re, a->re);
    cmul_tdvector(t2, val->val_im, a->im);
    cmul_tdvector(t3, val->val_im, a->re);
    cmul_tdvector(t4, val->val_re, a->im);

    sub_tdvector(c->re, t1, t2);
    add_tdvector(c->im, t3, t4);

    free_tdvector(t1);
    free_tdvector(t2);
    free_tdvector(t3);
    free_tdvector(t4);
}

/* c = val * a */
void cmul_ctdvector_3m(CTDVector c, ctdfloat *val, CTDVector a)
{
    TDVector t1, t2, t3;
    tdfloat tmp;

    t1 = init_tdvector(c->re->dim);
    t2 = init_tdvector(c->re->dim);

    cmul_tdvector(t1, val->val_re, a->re);
    cmul_tdvector(t2, val->val_im, a->im);
    sub_tdvector(c->re, t1, t2);

    //#ifdef USE_4M
    //    // 4M
    //    cmul_tdvector(t1, val->val_im, a->re);
    //    cmul_tdvector(t2, val->val_re, a->im);
    //    add_tdvector(c->im, t1, t2);
    //#else // USE_4M
        // 3M
        rtd_add(tmp.val, val->val_re, val->val_im);
        t3 = init_tdvector(c->re->dim);
        add_tdvector(t3, a->re, a->im);
        cmul_tdvector(c->im, tmp.val, t3);
        sub_tdvector(c->im, c->im, t1);
        sub_tdvector(c->im, c->im, t2);
        free_tdvector(t3);
    //#endif // USE_4M

    free_tdvector(t1);
    free_tdvector(t2);
}

/* c *= val */
void cmul2_ctdvector(CTDVector c, ctdfloat *val)
{
    CTDVector in_a;

    in_a = init_ctdvector(c->re->dim);

    subst_ctdvector(in_a, c);
    cmul_ctdvector(c, val, in_a);

    free_ctdvector(in_a);
}

/* c = a + val * b */
void add_cmul_ctdvector(CTDVector c, CTDVector a, ctdfloat *val, CTDVector b)
{
    CTDVector in_vec;
    in_vec = init_ctdvector(b->re->dim);

    //cmul_ctdvector(c, val, b);
    cmul_ctdvector(in_vec, val, b);
    //add2_ctdvector(c, a);
    add_ctdvector(c, a, in_vec);

    free_ctdvector(in_vec);
}

/* c = a - val * b */
void sub_cmul_ctdvector(CTDVector c, CTDVector a, ctdfloat *val, CTDVector b)
{
    CTDVector in_vec;
    in_vec = init_ctdvector(b->re->dim);

    //cmul_cddvector(c, val, b);
    cmul_ctdvector(in_vec, val, b);
    //sub2_cddvector(c, a);
    sub_ctdvector(c, a, in_vec);

    free_ctdvector(in_vec);
}


/* (a, b) = conj(a)^T * b */
void ip_ctdvector(ctdfloat *ret, CTDVector a, CTDVector b)
{
    int i;
    ctdfloat tmp, conj_a_i, ai, bi;

    rctd_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_ctdvector_i(&ai, a, i);
        subst_ctdvector_i(&bi, b, i);
        rctd_conj(&conj_a_i, &ai); // get_ctdvector_i(a, i));
        rctd_mul(&tmp, &conj_a_i, &bi); // get_ctdvector_i(b, i));
        rctd_add(ret, ret, &tmp);
    }
}

/* a^T * b */
void dotp_ctdvector(ctdfloat *ret, CTDVector a, CTDVector b)
{
    int i;
    ctdfloat tmp, ai, bi;

    rctd_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_ctdvector_i(&ai, a, i);
        subst_ctdvector_i(&bi, b, i);
        rctd_mul(&tmp, &ai, &bi); // get_ctdvector_i(a, i), get_ctdvector_i(b, i));
        rctd_add(ret, ret, &tmp);
    }
}

/* c := a */
void subst_ctdvector(CTDVector c, CTDVector a)
{
    subst_tdvector(c->re, a->re);
    subst_tdvector(c->im, a->im);
}

/* c := conj(a) */
void conj_ctdvector(CTDVector c, CTDVector a)
{
    subst_tdvector(c->re, a->re);
    neg_tdvector(c->im, a->im);
}

/* c := -a */
void neg_ctdvector(CTDVector c, CTDVector a)
{
    neg_tdvector(c->re, a->re);
    neg_tdvector(c->im, a->im);
}

/* ||a||_1 */
void norm1_ctdvector(double ret[TDSIZE], CTDVector a)
{
    int i;
    tdfloat tmp;
    ctdfloat ai;

    rtd_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_ctdvector_i(&ai, a, i);
        rctd_abs(&tmp, &ai); // get_ctdvector_i(a, i));
        rtd_add(ret, ret, tmp.val);
    }
}

/* ||a||_infty */
void normi_ctdvector(double ret[TDSIZE], CTDVector a)
{
    int i;
    tdfloat tmp;
    ctdfloat ai;

    rtd_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_ctdvector_i(&ai, a, i);
        rctd_abs(&tmp, &ai); // get_ctdvector_i(a, i));
        if(rtd_cmp(ret, tmp.val) < 0)
            rtd_set(ret, tmp.val);
    }
}

// Euclid norm
void norm2_ctdvector(double ret[TDSIZE], CTDVector vec)
{
    int i;
    tdfloat tmp;
    ctdfloat vec_i;

    rtd_set0(ret);
    for(i = 0; i < vec->re->dim; i++)
    {
        subst_ctdvector_i(&vec_i, vec, i);
        rctd_nrm2(&tmp, &vec_i); // get_ctdvector_i(vec, i));
        rtd_add(ret, ret, tmp.val);
    }
    #ifdef USE_GMP
    rtd_sqrt_mpfr(ret, ret);
    #else // USE_GMP
    rtd_sqrt(ret, ret);
    #endif // USE_GMP
}

// set a zero matrix
//void set0_ctdmatrix(CTDMatrix mat);
void set0_ctdmatrix(CTDMatrix mat)
{
    set0_tdmatrix(mat->re);
    set0_tdmatrix(mat->im);
}

// initialize tdvector
CTDMatrix init_ctdmatrix(long int row_dim, long int col_dim)
{
    CTDMatrix ret = NULL;

    ret = (CTDMatrix)malloc(sizeof(ctdmatrix));
    if(ret == NULL)
    {
        fprintf(stderr, "ERROR: init_ctdmatrix(%ld, %ld)\n", row_dim, col_dim);
        return NULL;
    }

    ret->re = init_tdmatrix(row_dim, col_dim);
    if(ret->re == NULL)
    {
        fprintf(stderr, "ERROR: init_ctdmatrix(%ld, %ld)\n", row_dim, col_dim);
        free(ret);
        return NULL;
    }

    ret->im = init_tdmatrix(row_dim, col_dim);
    if(ret->im == NULL)
    {
        fprintf(stderr, "ERROR: init_ctdmatrix(%ld, %ld)\n", row_dim, col_dim);
        free_tdmatrix(ret->re);
        free(ret);
        return NULL;
    }

    return ret;
}

// free cstdvector
void free_ctdmatrix(CTDMatrix mat)
{
    free_tdmatrix(mat->re);
    free_tdmatrix(mat->im);
    free(mat);
}

// print ctdvector
//void print_ctdmatrix(CTDMatrix mat);

// CTDMatrix mat -> ctdfloat array
void set_ctdfloat_ctdmat(ctdfloat ret[], int ret_dim, CTDMatrix mat)
{
    long int i, j, index;
    ctdfloat mat_ij;

    index = 0;
    for(i = 0; i < mat->re->row_dim; i++)
    {
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_ctdmatrix_ij(&mat_ij, mat, i, j);
            rctd_set(&ret[index++], &mat_ij); // get_ctdmatrix_ij(mat, i, j));
        }
    }

    return;
}

// tdmatrix -> ctdmatrix
void set_ctdmatrix_tdmat(CTDMatrix ret, TDMatrix re_mat, TDMatrix im_mat)
{
    subst_tdmatrix(ret->re, re_mat);
    subst_tdmatrix(ret->im, im_mat);
}

// tdfloat array -> DDmatrix ret
void set_ctdmatrix_ctdfloat(CTDMatrix ret, ctdfloat array[], int array_dim)
{
    long int i, j, index;

    index = 0;
    for(i = 0; i < ret->re->row_dim; i++)
    {
        for(j = 0; j < ret->re->col_dim; j++)
            set_ctdmatrix_ij(ret, i, j, &array[index++]);
    }

    return;
}

// matrix multiplication
// ret := A * B
void mul_ctdmatrix_4m(CTDMatrix ret, CTDMatrix a, CTDMatrix b)
{
    TDMatrix t1, t2, t3, t4;

    t1 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_tdmatrix(t1, a->re, b->re);
    mul_tdmatrix(t2, a->im, b->im);
    sub_tdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_tdmatrix(t3, a->im, b->re);
        mul_tdmatrix(t4, a->re, b->im);
        add_tdmatrix(ret->im, t3, t4);
    //#else // USE_4M 
        // 3M
    /*    add_tdmatrix(t3, a->re, a->im);
        add_tdmatrix(t4, b->re, b->im);
        mul_tdmatrix(ret->im, t3, t4);
        sub_tdmatrix(ret->im, ret->im, t1);
        sub_tdmatrix(ret->im, ret->im, t2);
    */
    //#endif // USE_4M

    free_tdmatrix(t1);
    free_tdmatrix(t2);
    free_tdmatrix(t3);
    free_tdmatrix(t4);
}

// matrix multiplication
// ret := A * B
void mul_ctdmatrix_3m(CTDMatrix ret, CTDMatrix a, CTDMatrix b)
{
    TDMatrix t1, t2, t3, t4;

    t1 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tdmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_tdmatrix(b->re->row_dim, ret->re->col_dim);

    mul_tdmatrix(t1, a->re, b->re);
    mul_tdmatrix(t2, a->im, b->im);
    sub_tdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
    /*
        mul_tdmatrix(t3, a->im, b->re);
        mul_tdmatrix(t4, a->re, b->im);
        add_tdmatrix(ret->im, t1, t2);
    */
    //#else // USE_4M 
        // 3M
        add_tdmatrix(t3, a->re, a->im);
        add_tdmatrix(t4, b->re, b->im);
        mul_tdmatrix(ret->im, t3, t4);
        sub_tdmatrix(ret->im, ret->im, t1);
        sub_tdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_tdmatrix(t1);
    free_tdmatrix(t2);
    free_tdmatrix(t3);
    free_tdmatrix(t4);
}

// Frobenius norm
void normf_ctdmatrix(double ret[TDSIZE], CTDMatrix mat)
{
    int i, j;
    tdfloat tmp;
    ctdfloat mat_ij;

    rtd_set0(ret);
    for(i = 0; i < mat->re->row_dim; i++)
    {
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_ctdmatrix_ij(&mat_ij, mat, i, j);
            rctd_nrm2(&tmp, &mat_ij); // get_ctdmatrix_ij(mat, i, j));
            rtd_add(ret, ret, tmp.val);
        }
    }
    rtd_sqrt(ret, ret);
}

// print normf
void print_normf_ctdmatrix(const char *str, CTDMatrix mat)
{
    tdfloat tmp;

    normf_ctdmatrix(tmp.val, mat);

    rtd_out_str(tmp.val);
}

/*************************************************/
/* Matrix Caluculations for CTDMatrix            */
/*
void normf_ctdmatrix(double ret[TDSIZE], CTDMatrix mat)
void norm1_ctdmatrix(double ret[TDSIZE], CTDMatrix mat)
void normi_ctdmatrix(double ret[TDSIZE], CTDMatrix mat)
void add_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b);
void sub_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b);
void mul_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b);
void mul_ctdmatrix_tdvec(CTDVector v, CTDMatrix a, CTDVector vb)
void mul_ctdmatrixt_tdvec(CTDVector v, CTDMatrix a, CTDVector vb)
void transpose_ctdmatrix(CTDMatrix c, CTDMatrix a);
void inv_ctdmatrix(CTDMatrix a);
void subst_mpfmatrux(CTDMatrix c, CTDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_ctdmatrix(double ret[TDSIZE], CTDMatrix mat)
{
    int i, j;
    tdfloat tmp, sum;
    ctdfloat mat_ij;

    rtd_set0(ret);
    for(j = 0; j < mat->re->col_dim; j++)
    {
        rtd_set0(sum.val);
        for(i = 0; i < mat->re->row_dim; i++)
        {
            subst_ctdmatrix_ij(&mat_ij, mat, i, j);
            rctd_abs(&tmp, &mat_ij); // get_ctdmatrix_ij(mat, i, j));
            rtd_add(sum.val, sum.val, tmp.val);
        }
        if(rtd_cmp(ret, sum.val) < 0)
            rtd_set(ret, sum.val);
    }
}

/* 1 Norm of Matrix */
void norm1_ctdmatrix(double ret[TDSIZE], CTDMatrix mat)
{
    int i, j;
    tdfloat tmp, sum;
    ctdfloat mat_ij;

    rtd_set0(ret);
    for(i = 0; i < mat->re->row_dim; i++)
    {
        rtd_set0(sum.val);
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_ctdmatrix_ij(&mat_ij, mat, i, j);
            rctd_abs(&tmp, &mat_ij); // get_ctdmatrix_ij(mat, i, j));
            rtd_add(sum.val, sum.val, tmp.val);
        }
        if(rtd_cmp(ret, sum.val) < 0)
            rtd_set(ret, sum.val);
    }
}

/* c := a + b */
void add_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b)
{
    add_tdmatrix(c->re, a->re, b->re);
    add_tdmatrix(c->im, a->im, b->im);
}

/* c := a - b */
void sub_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b)
{
    sub_tdmatrix(c->re, a->re, b->re);
    sub_tdmatrix(c->im, a->im, b->im);
}

/* c := sc * a */
void cmul_ctdmatrix_4m(CTDMatrix c, ctdfloat *sc, CTDMatrix a)
{
    TDMatrix t1, t2, t3;
    tdfloat tmp;

    t1 = init_tdmatrix(c->re->row_dim, c->re->col_dim);
    t2 = init_tdmatrix(c->re->row_dim, c->re->col_dim);

    cmul_tdmatrix(t1, sc->val_re, a->re);
    cmul_tdmatrix(t2, sc->val_im, a->im);
    sub_tdmatrix(c->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        cmul_tdmatrix(t1, sc->val_im, a->re);
        cmul_tdmatrix(t2, sc->val_re, a->im);
        add_tdmatrix(c->im, t1, t2);
    //#else // USE_4M
        // 3M
    /*
        rtd_add(tmp.val, sc->val_re, sc->val_im);
        t3 = init_tdmatrix(c->re->row_dim, c->im->col_dim);
        add_tdmatrix(t3, a->re, a->im);
        cmul_tdmatrix(c->im, tmp.val, t3);
        sub_tdmatrix(c->im, c->im, t1);
        sub_tdmatrix(c->im, c->im, t2);
        free_tdmatrix(t3);
    */
    //#endif // USE_4M

    free_tdmatrix(t1);
    free_tdmatrix(t2);
}

/* c := sc * a */
void cmul_ctdmatrix_3m(CTDMatrix c, ctdfloat *sc, CTDMatrix a)
{
    TDMatrix t1, t2, t3;
    tdfloat tmp;

    t1 = init_tdmatrix(c->re->row_dim, c->re->col_dim);
    t2 = init_tdmatrix(c->re->row_dim, c->re->col_dim);

    cmul_tdmatrix(t1, sc->val_re, a->re);
    cmul_tdmatrix(t2, sc->val_im, a->im);
    sub_tdmatrix(c->re, t1, t2);

    //#ifdef USE_4M
        // 4M
    /*
        cmul_tdmatrix(t1, sc->val_im, a->re);
        cmul_tdmatrix(t2, sc->val_re, a->im);
        add_tdmatrix(c->im, t1, t2);
    */
    //#else // USE_4M
        // 3M
        rtd_add(tmp.val, sc->val_re, sc->val_im);
        t3 = init_tdmatrix(c->re->row_dim, c->im->col_dim);
        add_tdmatrix(t3, a->re, a->im);
        cmul_tdmatrix(c->im, tmp.val, t3);
        sub_tdmatrix(c->im, c->im, t1);
        sub_tdmatrix(c->im, c->im, t2);
        free_tdmatrix(t3);
    //#endif // USE_4M

    free_tdmatrix(t1);
    free_tdmatrix(t2);
}

/* c = a^T */
void transpose_ctdmatrix(CTDMatrix c, CTDMatrix a)
{
    transpose_tdmatrix(c->re, a->re);
    transpose_tdmatrix(c->im, a->im);
}

/* c := conj(a)^T */
void star_ctdmatrix(CTDMatrix c, CTDMatrix a)
{
    // c_re := -a_im
    neg_tdmatrix(c->re, a->im);
    // c_im := -a_im^T
    transpose_tdmatrix(c->im, c->re);
    // c_re := a_re^T
    transpose_tdmatrix(c->re, a->re);
}

/* c := conj(a) */
void conj_ctdmatrix(CTDMatrix c, CTDMatrix a)
{
    subst_tdmatrix(c->re, a->re);
    neg_tdmatrix(c->im, a->im);
}

/* c := -a */
void neg_ctdmatrix(CTDMatrix c, CTDMatrix a)
{
    neg_tdmatrix(c->re, a->re);
    neg_tdmatrix(c->im, a->im);
}

/* c := a */
void subst_ctdmatrix(CTDMatrix c, CTDMatrix a)
{
    subst_tdmatrix(c->re, a->re);
    subst_tdmatrix(c->im, a->im);
}

/* c := I */
void setI_ctdmatrix(CTDMatrix c)
{
    setI_tdmatrix(c->re);
    set0_tdmatrix(c->im);
}

/* v := a * vb */
void mul_ctdmatrix_ctdvec_4m(CTDVector v, CTDMatrix a, CTDVector vb)
{
    TDVector t1, t2, t3, t4;
    TDMatrix tmp_mat;

    t1 = init_tdvector(v->re->dim);
    t2 = init_tdvector(v->re->dim);
    t3 = init_tdvector(v->re->dim);
    t4 = init_tdvector(v->re->dim);

    mul_tdmatrix_tdvec(t1, a->re, vb->re);
    mul_tdmatrix_tdvec(t2, a->im, vb->im);
    sub_tdvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        mul_tdmatrix_tdvec(t3, a->im, vb->re);
        mul_tdmatrix_tdvec(t4, a->re, vb->im);
        add_tdvector(v->im, t3, t4);
    //#else // USE_4M
        // 3M
    /*
        tmp_mat = init_tdmatrix(a->re->row_dim, a->re->col_dim);
        add_tdmatrix(tmp_mat, a->re, a->im);
        add_tdvector(t3, vb->re, vb->im);
        mul_tdmatrix_tdvec(t4, tmp_mat, t3);
        sub_tdvector(v->im, t4, t1);
        sub_tdvector(v->im, v->im, t2);
        free_tdmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_tdvector(t1);
    free_tdvector(t2);
    free_tdvector(t3);
    free_tdvector(t4);
}

/* v := a * vb */
void mul_ctdmatrix_ctdvec_3m(CTDVector v, CTDMatrix a, CTDVector vb)
{
    TDVector t1, t2, t3, t4;
    TDMatrix tmp_mat;

    t1 = init_tdvector(v->re->dim);
    t2 = init_tdvector(v->re->dim);
    t3 = init_tdvector(v->re->dim);
    t4 = init_tdvector(v->re->dim);

    mul_tdmatrix_tdvec(t1, a->re, vb->re);
    mul_tdmatrix_tdvec(t2, a->im, vb->im);
    sub_tdvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
    /*
        mul_tdmatrix_tdvec(t3, a->im, vb->re);
        mul_tdmatrix_tdvec(t4, a->re, vb->im);
        add_tdvector(v->im, t3, t4);
    */
    //#else // USE_4M
        // 3M
        tmp_mat = init_tdmatrix(a->re->row_dim, a->re->col_dim);
        add_tdmatrix(tmp_mat, a->re, a->im);
        add_tdvector(t3, vb->re, vb->im);
        mul_tdmatrix_tdvec(t4, tmp_mat, t3);
        sub_tdvector(v->im, t4, t1);
        sub_tdvector(v->im, v->im, t2);
        free_tdmatrix(tmp_mat);
    //#endif // USE_4M

    free_tdvector(t1);
    free_tdvector(t2);
    free_tdvector(t3);
    free_tdvector(t4);
}


/* v := a^T * vb */
void mul_ctdmatrixt_ctdvec(CTDVector v, CTDMatrix a, CTDVector vb)
{
    TDVector t1, t2, t3, t4;
    TDMatrix tmp_mat;

    t1 = init_tdvector(v->re->dim);
    t2 = init_tdvector(v->re->dim);
    t3 = init_tdvector(v->re->dim);
    t4 = init_tdvector(v->re->dim);

    mul_tdmatrixt_tdvec(t1, a->re, vb->re);
    mul_tdmatrixt_tdvec(t2, a->im, vb->im);
    sub_tdvector(v->re, t1, t2);

   // #ifdef USE_4M
        // 4M
        mul_tdmatrixt_tdvec(t3, a->im, vb->re);
        mul_tdmatrixt_tdvec(t4, a->re, vb->im);
        add_tdvector(v->im, t3, t4);
   // #else // USE_4M
        // 3M
        /*
        tmp_mat = init_tdmatrix(a->re->row_dim, a->re->col_dim);
        add_tdmatrix(tmp_mat, a->re, a->im);
        add_tdvector(t3, vb->re, vb->im);
        mul_tdmatrixt_tdvec(t4, tmp_mat, t3);
        sub_tdvector(v->im, t4, t1);
        sub_tdvector(v->im, v->im, t2);
        free_tdmatrix(tmp_mat);
        */
    //#endif // USE_4M

    free_tdvector(t1);
    free_tdvector(t2);
    free_tdvector(t3);
    free_tdvector(t4);
}

/* v := conj(a)^T * vb */
void mul_ctdmatrixs_ctdvec(CTDVector v, CTDMatrix a, CTDVector vb)
{
    TDVector t1, t2, t3, t4;
    TDMatrix tmp_mat;

    t1 = init_tdvector(v->re->dim);
    t2 = init_tdvector(v->re->dim);
    t3 = init_tdvector(v->re->dim);
    t4 = init_tdvector(v->re->dim);

    mul_tdmatrixt_tdvec(t1, a->re, vb->re);
    mul_tdmatrixt_tdvec(t2, a->im, vb->im);
    add_tdvector(v->re, t1, t2);

   // #ifdef USE_4M
        // 4M
        mul_tdmatrixt_tdvec(t3, a->im, vb->re);
        mul_tdmatrixt_tdvec(t4, a->re, vb->im);
        sub_tdvector(v->im, t3, t4);
   // #else // USE_4M
        // 3M
        /*
        tmp_mat = init_tdmatrix(a->re->row_dim, a->re->col_dim);
        add_tdmatrix(tmp_mat, a->re, a->im);
        add_tdvector(t3, vb->re, vb->im);
        mul_tdmatrixt_tdvec(t4, tmp_mat, t3);
        sub_tdvector(v->im, t4, t1);
        sub_tdvector(v->im, v->im, t2);
        free_tdmatrix(tmp_mat);
        */
    //#endif // USE_4M

    free_tdvector(t1);
    free_tdvector(t2);
    free_tdvector(t3);
    free_tdvector(t4);
}


/* a = a^(-1) */
/* square matrix only */
void inv_ctdmatrix(CTDMatrix a)
{
	long int i, j, k, dim;
	ctdfloat ctmp, aii, aij, aik, ajk, aji;
	tdfloat tmp;

	/* Check Dimensions */
	if(a->re->row_dim != a->re->col_dim)
	{
		fprintf(stderr, "ERROR: inv_ctdmatrix\n");
		return;
	}

	dim = a->re->row_dim;

	for(i = 0; i < dim; i++)
	{
        subst_ctdmatrix_ij(&ctmp, a, i, i);
		rctd_abs(&tmp, &ctmp); // get_ctdmatrix_ij(a, i, i));
		if(rtd_cmp_ui(tmp.val, 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_ctdmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		rctd_inv(&aii, &ctmp); // get_ctdmatrix_ij(a, i, i));
		//inv_mpc_t(aii, get_cmpfmatrix_ij(a, i, i));
		set_ctdmatrix_ij(a, i, i, &aii);

		for(j = 0; j < i; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
            subst_ctdmatrix_ij(&aij, a, i, j);
			rctd_mul(&ctmp, &aij, &aii); // get_ctdmatrix_ij(a, i, j), &aii);
			set_ctdmatrix_ij(a, i, j, &ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
            subst_ctdmatrix_ij(&aij, a, i, j);
			rctd_mul(&ctmp, &aij, &aii); // get_ctdmatrix_ij(a, i, j), &aii);
			set_ctdmatrix_ij(a, i, j, &ctmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_ctdmatrix_ij(&aji, a, j, i);
                subst_ctdmatrix_ij(&aik, a, i, k);
                subst_ctdmatrix_ij(&ajk, a, j, k);
				rctd_mul(&ctmp, &aji, &aik); // get_ctdmatrix_ij(a, j, i), get_ctdmatrix_ij(a, i, k));
				rctd_sub(&ctmp, &ajk, &ctmp); // get_ctdmatrix_ij(a, j, k), &ctmp);
				set_ctdmatrix_ij(a, j, k, &ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_ctdmatrix_ij(&aji, a, j, i);
                subst_ctdmatrix_ij(&aik, a, i, k);
                subst_ctdmatrix_ij(&ajk, a, j, k);
				rctd_mul(&ctmp, &aji, &aik); // get_ctdmatrix_ij(a, j, i), get_ctdmatrix_ij(a, i, k));
				rctd_sub(&ctmp, &ajk, &ctmp); // get_ctdmatrix_ij(a, j, k), &ctmp);
				set_ctdmatrix_ij(a, j, k, &ctmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_ctdmatrix_ij(&aji, a, j, i);
                subst_ctdmatrix_ij(&aik, a, i, k);
                subst_ctdmatrix_ij(&ajk, a, j, k);
				rctd_mul(&ctmp, &aji, &aik); // get_ctdmatrix_ij(a, j, i), get_ctdmatrix_ij(a, i, k));
				rctd_sub(&ctmp, &ajk, &ctmp); // get_ctdmatrix_ij(a, j, k), &ctmp);
				set_ctdmatrix_ij(a, j, k, &ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_ctdmatrix_ij(&aji, a, j, i);
                subst_ctdmatrix_ij(&aik, a, i, k);
                subst_ctdmatrix_ij(&ajk, a, j, k);
				rctd_mul(&ctmp, &aji, &aik); // get_ctdmatrix_ij(a, j, i), get_ctdmatrix_ij(a, i, k));
				rctd_sub(&ctmp, &ajk, &ctmp); // get_ctdmatrix_ij(a, j, k), &ctmp);
				set_ctdmatrix_ij(a, j, k, &ctmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			rctd_neg(&ctmp, &aii);
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
            subst_ctdmatrix_ij(&aji, a, j, i);
			rctd_mul(&ctmp, &ctmp, &aji); // get_ctdmatrix_ij(a, j, i));
			set_ctdmatrix_ij(a, j, i, &ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			rctd_neg(&ctmp, &aii);
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
            subst_ctdmatrix_ij(&aji, a, j, i);
            rctd_mul(&ctmp, &ctmp, &aji); // get_ctdmatrix_ij(a, j, i));
			set_ctdmatrix_ij(a, j, i, &ctmp);
		}
	}
}

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpc)a */
void subst_cmpfvector_ctdvec(CMPFVector c, CTDVector a)
{
    MPFVector re, im;

    re = init2_mpfvector(c->dim, c->prec);
    im = init2_mpfvector(c->dim, c->prec);

    subst_mpfvector_tdvec(re, a->re);
    subst_mpfvector_tdvec(im, a->im);

    merge_cmpfvector(c, re, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (ctd)a */
void subst_ctdvector_cmpfvec(CTDVector c, CMPFVector a)
{
    MPFVector re, im;

    re = init2_mpfvector(a->dim, a->prec);
    im = init2_mpfvector(a->dim, a->prec);

    separate_cmpfvector(re, im, a);
    subst_tdvector_mpfvec(c->re, re);
    subst_tdvector_mpfvec(c->im, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (mpf)a */
void subst_cmpfmatrix_ctdmat(CMPFMatrix c, CTDMatrix a)
{
    MPFMatrix re, im;

    re = init2_mpfmatrix(c->row_dim, c->col_dim, c->prec);
    im = init2_mpfmatrix(c->row_dim, c->col_dim, c->prec);

    subst_mpfmatrix_tdmat(re, a->re);
    subst_mpfmatrix_tdmat(im, a->im);

    merge_cmpfmatrix(c, re, im);

    free_mpfmatrix(re);
    free_mpfmatrix(im);
}


/* c := (ctd)a */
void subst_ctdmatrix_cmpfmat(CTDMatrix c, CMPFMatrix a)
{
    MPFMatrix re, im;

    re = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    im = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);

    separate_cmpfmatrix(re, im, a);
    subst_tdmatrix_mpfmat(c->re, re);
    subst_tdmatrix_mpfmat(c->im, im);

    free_mpfmatrix(re);
    free_mpfmatrix(im);
}

/* Normwise relative error of vector */
void relerr_ctdvector_cmpfvec(double relerr[TDSIZE], CTDVector approx_vec, CMPFVector true_vec, int norm_type)
{}

/* Elementwise relative errors of vector */
void relerr_element_ctdvector_mpf(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], CTDVector approx_vec, MPFVector true_vec, int norm_type)
{}
#endif // USE_GMP

/* c := (td)a */
void subst_ctdvector_cdvec(CTDVector c, CDVector a)
{
    DVector re, im;

    re = init_dvector(a->dim);
    im = init_dvector(a->dim);

    separate_cdvector(re, im, a);
    subst_tdvector_dvec(c->re, re);
    subst_tdvector_dvec(c->im, im);

    free_dvector(re);
    free_dvector(im);
}

/* c := (d)a */
void subst_cdvector_ctdvec(CDVector c, CTDVector a)
{
    DVector re, im;

    re = init_dvector(a->re->dim);
    im = init_dvector(a->im->dim);

    subst_dvector_tdvec(re, a->re);
    subst_dvector_tdvec(im, a->im);

    merge_cdvector(c, re, im);

    free_dvector(re);
    free_dvector(im);
}

/* c := (td)a */
void subst_ctdmatrix_cdmat(CTDMatrix c, CDMatrix a)
{
    DMatrix re, im;

    re = init_dmatrix(a->row_dim, a->col_dim);
    im = init_dmatrix(a->row_dim, a->col_dim);

    separate_cdmatrix(re, im, a);
    subst_tdmatrix_dmat(c->re, re);
    subst_tdmatrix_dmat(c->im, im);

    free_dmatrix(re);
    free_dmatrix(im);
}

/* c := (d)a */
void subst_cdmatrix_ctdmat(CDMatrix c, CTDMatrix a)
{
    DMatrix re, im;

    re = init_dmatrix(c->row_dim, c->col_dim);
    im = init_dmatrix(c->row_dim, c->col_dim);

    subst_dmatrix_tdmat(re, a->re);
    subst_dmatrix_tdmat(im, a->im);
    merge_cdmatrix(c, re, im);

    free_dmatrix(re);
    free_dmatrix(im);
}

#ifdef USE_DDLINEAR

/* c := (td)a */
void subst_ctdvector_cddvec(CTDVector c, CDDVector a)
{
    subst_tdvector_ddvec(c->re, a->re);
    subst_tdvector_ddvec(c->im, a->im);
}

/* c := (dd)a */
void subst_cddvector_ctdvec(CDDVector c, CTDVector a)
{
    subst_ddvector_tdvec(c->re, a->re);
    subst_ddvector_tdvec(c->im, a->im);
}

/* c := (td)a */
void subst_ctdmatrix_cddmat(CTDMatrix c, CDDMatrix a)
{
    subst_tdmatrix_ddmat(c->re, a->re);
    subst_tdmatrix_ddmat(c->im, a->im);
}

/* c := (dd)a */
void subst_cddmatrix_ctdmat(CDDMatrix c, CTDMatrix a)
{
    subst_ddmatrix_tdmat(c->re, a->re);
    subst_ddmatrix_tdmat(c->im, a->im);
}

#endif // USE_DDLINEAR

/* Normwise relative error of vector */
void relerr_ctdvector(double relerr[TDSIZE], CTDVector approx_vec, CTDVector true_vec, int norm_type)
{
    double norm_true_vec[TDSIZE], norm_diff_vec[TDSIZE];
	CTDVector diff_vec;

	diff_vec = init_ctdvector(approx_vec->re->dim);

	// diff_vec := approx_vec - true_vec
	sub_ctdvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_ctdvector(norm_diff_vec, diff_vec);
			normi_ctdvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_ctdvector(norm_diff_vec, diff_vec);
			norm1_ctdvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_ctdvector(norm_diff_vec, diff_vec);
			norm2_ctdvector(norm_true_vec, true_vec);
			break;
	}

	if(rtd_cmp_ui(norm_true_vec, 0UL) != 0)
		rtd_div(relerr, norm_diff_vec, norm_true_vec);

	free_ctdvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_ctdvector(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], CTDVector approx_vec, CTDVector true_vec, int norm_type)
{
    tdfloat abs_true_vec, abs_diff_vec;
    double norm_diff_vec[TDSIZE], norm_true_vec[TDSIZE];
	long int i;
	CTDVector diff_vec;
    ctdfloat ctmp;

	diff_vec = init_ctdvector(approx_vec->re->dim);

	// diff_vec := approx_vec - true_vec
	sub_ctdvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_ctdvector(norm_diff_vec, diff_vec);
			normi_ctdvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_ctdvector(norm_diff_vec, diff_vec);
			norm1_ctdvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_ctdvector(norm_diff_vec, diff_vec);
			norm2_ctdvector(norm_true_vec, true_vec);
			break;
	}

	rtd_set(norm_relerr, norm_diff_vec);
	if(rtd_cmp_ui(norm_true_vec, 0UL) != 0)
		rtd_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rtd_set_ui(max_relerr, 0UL);
	normi_ctdvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->re->dim; i++)
	{
        subst_ctdvector_i(&ctmp, diff_vec, i);
		rctd_abs(&abs_diff_vec, &ctmp); // get_ctdvector_i(diff_vec, i));
        subst_ctdvector_i(&ctmp, true_vec, i);
		rctd_abs(&abs_true_vec, &ctmp); // get_ctdvector_i(true_vec, i));
		if(rtd_cmp_ui(abs_true_vec.val, 0UL) != 0)
			rtd_div(abs_diff_vec.val, abs_diff_vec.val, abs_true_vec.val);
		
		if(rtd_cmp(max_relerr, abs_diff_vec.val) < 0)
			rtd_set(max_relerr, abs_diff_vec.val);
		if(rtd_cmp(min_relerr, abs_diff_vec.val) > 0)
			rtd_set(min_relerr, abs_diff_vec.val);
	}

	free_ctdvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_ctdmatrix(CTDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
    row_swap_tdmatrix(mat->re, row_index0, row_index1, col_start, col_end);
    row_swap_tdmatrix(mat->im, row_index0, row_index1, col_start, col_end);
}

// print ctdmatrix
void print_ctdmatrix(CTDMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->re->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->re->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_dd_write((vec->element + index * TDSIZE));
			//c_dd_write(GET_DDMATRIX_IJ(mat, row_index, col_index));
			rtd_out_str(get_ctdmatrix_ij_ctdfloat(mat, row_index, col_index).val_re);
            printf(" + ");
   			rtd_out_str(get_ctdmatrix_ij_ctdfloat(mat, row_index, col_index).val_im);
            printf(" * I\n");
		}
	}
}


#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus

#ifdef DEBUG
int main()
{
    tdfloat a, b, c
    ctdfloat ca, cb, cc;

    rtd_set_d(a->val, 3.0); rtd_sqrt(a->val);
    rtd_set_d(b->val, 5.0); rtd_sqrt(b->val);

    rtd_out_str(a);
    rtd_out_str(b);
    
}
#endif // DEBUG

