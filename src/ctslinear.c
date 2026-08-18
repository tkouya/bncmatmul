/********************************************************************************/
/* ctdlinear.c: Triple-float precision Complex Linear Computation Library      */
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
#include "ctslinear.h"
//#include "clinear.h"

#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dtq_sd.h"
#include "mpflinear.h"
#endif //USE_GMP//

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// file-local: real ts matrix negation (no neg_tsmatrix in single-precision lib)
static void _cts_neg_tsmatrix(TSMatrix c, TSMatrix a)
{
    float neg_one[TSSIZE];
    int i;
    neg_one[0] = -1.0f; for(i = 1; i < TSSIZE; i++) neg_one[i] = 0.0f;
    cmul_tsmatrix(c, neg_one, a);
}

// initialize CTSVector
CTSVector init_ctsvector(int dimension)
{
    CTSVector ret = NULL;

    ret = (CTSVector)malloc(sizeof(ctsvector));

    if(ret == NULL)
    {
        fprintf(stderr, "ERROR: init_ctsvector(%d)\n", dimension);
        return NULL;
    }

    ret->re = init_tsvector(dimension);
    if(ret->re == NULL)
    {
        fprintf(stderr, "ERROR: init_ctsvector(%d)\n", dimension);
        free(ret);
        return NULL;
    }

    ret->im = init_tsvector(dimension);
    if(ret->im == NULL)
    {
        fprintf(stderr, "ERROR: init_ctsvector(%d)\n", dimension);
        free_tsvector(ret->re);
        free(ret);
        return NULL;
    }

    return ret;
}

// free CTSVector
void free_ctsvector(CTSVector vec)
{
    free_tsvector(vec->re);
    free_tsvector(vec->im);
    free(vec);
}

// CTSVector vec -> tsfloat array
void set_ctsfloat_ctsvec(ctsfloat ret[], int ret_dim, CTSVector vec)
{
    int i;
    ctsfloat ptr_val;

    for(i = 0; i < ret_dim; i++)
    {
        subst_ctsvector_i(&ptr_val, vec, i);
        rcts_set(&ret[i], &ptr_val); // get_ctsvector_i(vec, i));
    }

/*    {
        ptr_val = get_ctsvector_i_ctsfloat(vec, i);
        rcts_set(&ret[i], &ptr_val);
    }
*/

    return;
}

// ctsfloat array -> CTSVector ret
void set_ctsvector_ctsfloat(CTSVector ret, ctsfloat array[], int array_dim)
{
    int i;

    for(i = 0; i < ret->re->dim; i++)
        set_ctsvector_i(ret, i, &array[i]);
}

// tsvector -> ctsvector
void set_ctsvector_tsvec(CTSVector ret, TSVector re_vec, TSVector im_vec)
{
    subst_tsvector(ret->re, re_vec);
    subst_tsvector(ret->im, im_vec);
}

// print tsvector
void print_ctsvector(CTSVector vec)
{
    int i;

    for(i = 0; i < vec->re->dim; i++)
    {
        printf("%5d ", i);
        rts_out_str_base(stdout, 10, 48, get_tsvector_i(vec->re, i));
        printf(" + ");
        rts_out_str_base(stdout, 10, 48, get_tsvector_i(vec->im, i));
        printf("\n");
    }
}

// set a zero vector
void set0_ctsvector(CTSVector vec)
{
    set0_tsvector(vec->re);
    set0_tsvector(vec->im);
}

/*************************************************/
/* Vector Calculations for CTSVector               */
/*
void add_ctsvector(CTSVector c, CTSVector a, CTSVector b)
void add2_ctsvector(CTSVector c, CTSVector a)
void sub_ctsvector(CTSVector c, CTSVector a, CTSVector b)
void sub2_ctsvector(CTSVector c, DVector a)
void cmul_ctsvector(CTSVector c, float val[TSSIZE], CTSVector a)
void cmul2_ctsvector(CTSVector c, float val[TSSIZE])
void add_cmul_ctsvector(CTSVector c, CTSVector a, float val[TSSIZE], CTSVector b)
float ip_ctsvector(CTSVector a, CTSVector b)
float norm1_ctsvector(CTSVector a)
float norm2_ctsvector(CTSVector a)
float normi_ctsvector(CTSVector a)
void subst_ctsvector(CTSVector c, CTSVector a)
*/
/*************************************************/
/* c = a + b */
void add_ctsvector(CTSVector c, CTSVector a, CTSVector b)
{
    add_tsvector(c->re, a->re, b->re);
    add_tsvector(c->im, a->im, b->im);
}

/* c += a */
void add2_ctsvector(CTSVector c, CTSVector a)
{
    add2_tsvector(c->re, a->re);
    add2_tsvector(c->im, a->im);
}

/* c = a - b */
void sub_ctsvector(CTSVector c, CTSVector a, CTSVector b)
{
    sub_tsvector(c->re, a->re, b->re);
    sub_tsvector(c->im, a->im, b->im);
}

/* c -= a */
void sub2_ctsvector(CTSVector c, CTSVector a)
{
    sub2_tsvector(c->re, a->re);
    sub2_tsvector(c->im, a->im);
}

/* c = val * a */
void cmul_ctsvector_4m(CTSVector c, ctsfloat *val, CTSVector a)
{
    TSVector t1, t2, t3, t4;

    // 2024-11-28(Thu) Fixed! T.Kouya
    t1 = init_tsvector(c->re->dim);
    t2 = init_tsvector(c->re->dim);
    t3 = init_tsvector(c->re->dim);
    t4 = init_tsvector(c->re->dim);

    cmul_tsvector(t1, val->val_re, a->re);
    cmul_tsvector(t2, val->val_im, a->im);
    cmul_tsvector(t3, val->val_im, a->re);
    cmul_tsvector(t4, val->val_re, a->im);

    sub_tsvector(c->re, t1, t2);
    add_tsvector(c->im, t3, t4);

    free_tsvector(t1);
    free_tsvector(t2);
    free_tsvector(t3);
    free_tsvector(t4);
}

/* c = val * a */
void cmul_ctsvector_3m(CTSVector c, ctsfloat *val, CTSVector a)
{
    TSVector t1, t2, t3;
    tsfloat tmp;

    t1 = init_tsvector(c->re->dim);
    t2 = init_tsvector(c->re->dim);

    cmul_tsvector(t1, val->val_re, a->re);
    cmul_tsvector(t2, val->val_im, a->im);
    sub_tsvector(c->re, t1, t2);

    //#ifdef USE_4M
    //    // 4M
    //    cmul_tsvector(t1, val->val_im, a->re);
    //    cmul_tsvector(t2, val->val_re, a->im);
    //    add_tsvector(c->im, t1, t2);
    //#else // USE_4M
        // 3M
        rts_add(tmp.val, val->val_re, val->val_im);
        t3 = init_tsvector(c->re->dim);
        add_tsvector(t3, a->re, a->im);
        cmul_tsvector(c->im, tmp.val, t3);
        sub_tsvector(c->im, c->im, t1);
        sub_tsvector(c->im, c->im, t2);
        free_tsvector(t3);
    //#endif // USE_4M

    free_tsvector(t1);
    free_tsvector(t2);
}

/* c *= val */
void cmul2_ctsvector(CTSVector c, ctsfloat *val)
{
    CTSVector in_a;

    in_a = init_ctsvector(c->re->dim);

    subst_ctsvector(in_a, c);
    cmul_ctsvector(c, val, in_a);

    free_ctsvector(in_a);
}

/* c = a + val * b */
void add_cmul_ctsvector(CTSVector c, CTSVector a, ctsfloat *val, CTSVector b)
{
    CTSVector in_vec;
    in_vec = init_ctsvector(b->re->dim);

    //cmul_ctsvector(c, val, b);
    cmul_ctsvector(in_vec, val, b);
    //add2_ctsvector(c, a);
    add_ctsvector(c, a, in_vec);

    free_ctsvector(in_vec);
}

/* c = a - val * b */
void sub_cmul_ctsvector(CTSVector c, CTSVector a, ctsfloat *val, CTSVector b)
{
    CTSVector in_vec;
    in_vec = init_ctsvector(b->re->dim);

    //cmul_cdsvector(c, val, b);
    cmul_ctsvector(in_vec, val, b);
    //sub2_cdsvector(c, a);
    sub_ctsvector(c, a, in_vec);

    free_ctsvector(in_vec);
}


/* (a, b) = conj(a)^T * b */
void ip_ctsvector(ctsfloat *ret, CTSVector a, CTSVector b)
{
    int i;
    ctsfloat tmp, conj_a_i, ai, bi;

    rcts_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_ctsvector_i(&ai, a, i);
        subst_ctsvector_i(&bi, b, i);
        rcts_conj(&conj_a_i, &ai); // get_ctsvector_i(a, i));
        rcts_mul(&tmp, &conj_a_i, &bi); // get_ctsvector_i(b, i));
        rcts_add(ret, ret, &tmp);
    }
}

/* a^T * b */
void dotp_ctsvector(ctsfloat *ret, CTSVector a, CTSVector b)
{
    int i;
    ctsfloat tmp, ai, bi;

    rcts_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_ctsvector_i(&ai, a, i);
        subst_ctsvector_i(&bi, b, i);
        rcts_mul(&tmp, &ai, &bi); // get_ctsvector_i(a, i), get_ctsvector_i(b, i));
        rcts_add(ret, ret, &tmp);
    }
}

/* c := a */
void subst_ctsvector(CTSVector c, CTSVector a)
{
    subst_tsvector(c->re, a->re);
    subst_tsvector(c->im, a->im);
}

/* c := conj(a) */
void conj_ctsvector(CTSVector c, CTSVector a)
{
    subst_tsvector(c->re, a->re);
    neg_tsvector(c->im, a->im);
}

/* c := -a */
void neg_ctsvector(CTSVector c, CTSVector a)
{
    neg_tsvector(c->re, a->re);
    neg_tsvector(c->im, a->im);
}

/* ||a||_1 */
void norm1_ctsvector(float ret[TSSIZE], CTSVector a)
{
    int i;
    tsfloat tmp;
    ctsfloat ai;

    rts_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_ctsvector_i(&ai, a, i);
        rcts_abs(&tmp, &ai); // get_ctsvector_i(a, i));
        rts_add(ret, ret, tmp.val);
    }
}

/* ||a||_infty */
void normi_ctsvector(float ret[TSSIZE], CTSVector a)
{
    int i;
    tsfloat tmp;
    ctsfloat ai;

    rts_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_ctsvector_i(&ai, a, i);
        rcts_abs(&tmp, &ai); // get_ctsvector_i(a, i));
        if(rts_cmp(ret, tmp.val) < 0)
            rts_set(ret, tmp.val);
    }
}

// Euclid norm
void norm2_ctsvector(float ret[TSSIZE], CTSVector vec)
{
    int i;
    tsfloat tmp;
    ctsfloat vec_i;

    rts_set0(ret);
    for(i = 0; i < vec->re->dim; i++)
    {
        subst_ctsvector_i(&vec_i, vec, i);
        rcts_nrm2(&tmp, &vec_i); // get_ctsvector_i(vec, i));
        rts_add(ret, ret, tmp.val);
    }
    #ifdef USE_GMP
    rts_sqrt_mpfr(ret, ret);
    #else // USE_GMP
    rts_sqrt(ret, ret);
    #endif // USE_GMP
}

// set a zero matrix
//void set0_ctsmatrix(CTSMatrix mat);
void set0_ctsmatrix(CTSMatrix mat)
{
    set0_tsmatrix(mat->re);
    set0_tsmatrix(mat->im);
}

// initialize tsvector
CTSMatrix init_ctsmatrix(long int row_dim, long int col_dim)
{
    CTSMatrix ret = NULL;

    ret = (CTSMatrix)malloc(sizeof(ctsmatrix));
    if(ret == NULL)
    {
        fprintf(stderr, "ERROR: init_ctsmatrix(%ld, %ld)\n", row_dim, col_dim);
        return NULL;
    }

    ret->re = init_tsmatrix(row_dim, col_dim);
    if(ret->re == NULL)
    {
        fprintf(stderr, "ERROR: init_ctsmatrix(%ld, %ld)\n", row_dim, col_dim);
        free(ret);
        return NULL;
    }

    ret->im = init_tsmatrix(row_dim, col_dim);
    if(ret->im == NULL)
    {
        fprintf(stderr, "ERROR: init_ctsmatrix(%ld, %ld)\n", row_dim, col_dim);
        free_tsmatrix(ret->re);
        free(ret);
        return NULL;
    }

    return ret;
}

// free cstsvector
void free_ctsmatrix(CTSMatrix mat)
{
    free_tsmatrix(mat->re);
    free_tsmatrix(mat->im);
    free(mat);
}

// print ctsvector
//void print_ctsmatrix(CTSMatrix mat);

// CTSMatrix mat -> ctsfloat array
void set_ctsfloat_ctsmat(ctsfloat ret[], int ret_dim, CTSMatrix mat)
{
    long int i, j, index;
    ctsfloat mat_ij;

    index = 0;
    for(i = 0; i < mat->re->row_dim; i++)
    {
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_ctsmatrix_ij(&mat_ij, mat, i, j);
            rcts_set(&ret[index++], &mat_ij); // get_ctsmatrix_ij(mat, i, j));
        }
    }

    return;
}

// tsmatrix -> ctsmatrix
void set_ctsmatrix_tsmat(CTSMatrix ret, TSMatrix re_mat, TSMatrix im_mat)
{
    subst_tsmatrix(ret->re, re_mat);
    subst_tsmatrix(ret->im, im_mat);
}

// tsfloat array -> DDmatrix ret
void set_ctsmatrix_ctsfloat(CTSMatrix ret, ctsfloat array[], int array_dim)
{
    long int i, j, index;

    index = 0;
    for(i = 0; i < ret->re->row_dim; i++)
    {
        for(j = 0; j < ret->re->col_dim; j++)
            set_ctsmatrix_ij(ret, i, j, &array[index++]);
    }

    return;
}

// matrix multiplication
// ret := A * B
void mul_ctsmatrix_4m(CTSMatrix ret, CTSMatrix a, CTSMatrix b)
{
    TSMatrix t1, t2, t3, t4;

    t1 = init_tsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tsmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_tsmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_tsmatrix(t1, a->re, b->re);
    mul_tsmatrix(t2, a->im, b->im);
    sub_tsmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_tsmatrix(t3, a->im, b->re);
        mul_tsmatrix(t4, a->re, b->im);
        add_tsmatrix(ret->im, t3, t4);
    //#else // USE_4M 
        // 3M
    /*    add_tsmatrix(t3, a->re, a->im);
        add_tsmatrix(t4, b->re, b->im);
        mul_tsmatrix(ret->im, t3, t4);
        sub_tsmatrix(ret->im, ret->im, t1);
        sub_tsmatrix(ret->im, ret->im, t2);
    */
    //#endif // USE_4M

    free_tsmatrix(t1);
    free_tsmatrix(t2);
    free_tsmatrix(t3);
    free_tsmatrix(t4);
}

// matrix multiplication
// ret := A * B
void mul_ctsmatrix_3m(CTSMatrix ret, CTSMatrix a, CTSMatrix b)
{
    TSMatrix t1, t2, t3, t4;

    t1 = init_tsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tsmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_tsmatrix(b->re->row_dim, ret->re->col_dim);

    mul_tsmatrix(t1, a->re, b->re);
    mul_tsmatrix(t2, a->im, b->im);
    sub_tsmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
    /*
        mul_tsmatrix(t3, a->im, b->re);
        mul_tsmatrix(t4, a->re, b->im);
        add_tsmatrix(ret->im, t1, t2);
    */
    //#else // USE_4M 
        // 3M
        add_tsmatrix(t3, a->re, a->im);
        add_tsmatrix(t4, b->re, b->im);
        mul_tsmatrix(ret->im, t3, t4);
        sub_tsmatrix(ret->im, ret->im, t1);
        sub_tsmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_tsmatrix(t1);
    free_tsmatrix(t2);
    free_tsmatrix(t3);
    free_tsmatrix(t4);
}

// Frobenius norm
void normf_ctsmatrix(float ret[TSSIZE], CTSMatrix mat)
{
    int i, j;
    tsfloat tmp;
    ctsfloat mat_ij;

    rts_set0(ret);
    for(i = 0; i < mat->re->row_dim; i++)
    {
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_ctsmatrix_ij(&mat_ij, mat, i, j);
            rcts_nrm2(&tmp, &mat_ij); // get_ctsmatrix_ij(mat, i, j));
            rts_add(ret, ret, tmp.val);
        }
    }
    rts_sqrt(ret, ret);
}

// print normf
void print_normf_ctsmatrix(const char *str, CTSMatrix mat)
{
    tsfloat tmp;

    normf_ctsmatrix(tmp.val, mat);

    rts_out_str_base(stdout, 10, 48, tmp.val);
}

/*************************************************/
/* Matrix Caluculations for CTSMatrix            */
/*
void normf_ctsmatrix(float ret[TSSIZE], CTSMatrix mat)
void norm1_ctsmatrix(float ret[TSSIZE], CTSMatrix mat)
void normi_ctsmatrix(float ret[TSSIZE], CTSMatrix mat)
void add_ctsmatrix(CTSMatrix c, CTSMatrix a, CTSMatrix b);
void sub_ctsmatrix(CTSMatrix c, CTSMatrix a, CTSMatrix b);
void mul_ctsmatrix(CTSMatrix c, CTSMatrix a, CTSMatrix b);
void mul_ctsmatrix_tsvec(CTSVector v, CTSMatrix a, CTSVector vb)
void mul_ctsmatrixt_tsvec(CTSVector v, CTSMatrix a, CTSVector vb)
void transpose_ctsmatrix(CTSMatrix c, CTSMatrix a);
void inv_ctsmatrix(CTSMatrix a);
void subst_mpfmatrux(CTSMatrix c, CTSMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_ctsmatrix(float ret[TSSIZE], CTSMatrix mat)
{
    int i, j;
    tsfloat tmp, sum;
    ctsfloat mat_ij;

    rts_set0(ret);
    for(j = 0; j < mat->re->col_dim; j++)
    {
        rts_set0(sum.val);
        for(i = 0; i < mat->re->row_dim; i++)
        {
            subst_ctsmatrix_ij(&mat_ij, mat, i, j);
            rcts_abs(&tmp, &mat_ij); // get_ctsmatrix_ij(mat, i, j));
            rts_add(sum.val, sum.val, tmp.val);
        }
        if(rts_cmp(ret, sum.val) < 0)
            rts_set(ret, sum.val);
    }
}

/* 1 Norm of Matrix */
void norm1_ctsmatrix(float ret[TSSIZE], CTSMatrix mat)
{
    int i, j;
    tsfloat tmp, sum;
    ctsfloat mat_ij;

    rts_set0(ret);
    for(i = 0; i < mat->re->row_dim; i++)
    {
        rts_set0(sum.val);
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_ctsmatrix_ij(&mat_ij, mat, i, j);
            rcts_abs(&tmp, &mat_ij); // get_ctsmatrix_ij(mat, i, j));
            rts_add(sum.val, sum.val, tmp.val);
        }
        if(rts_cmp(ret, sum.val) < 0)
            rts_set(ret, sum.val);
    }
}

/* c := a + b */
void add_ctsmatrix(CTSMatrix c, CTSMatrix a, CTSMatrix b)
{
    add_tsmatrix(c->re, a->re, b->re);
    add_tsmatrix(c->im, a->im, b->im);
}

/* c := a - b */
void sub_ctsmatrix(CTSMatrix c, CTSMatrix a, CTSMatrix b)
{
    sub_tsmatrix(c->re, a->re, b->re);
    sub_tsmatrix(c->im, a->im, b->im);
}

/* c := sc * a */
void cmul_ctsmatrix_4m(CTSMatrix c, ctsfloat *sc, CTSMatrix a)
{
    TSMatrix t1, t2, t3;
    tsfloat tmp;

    t1 = init_tsmatrix(c->re->row_dim, c->re->col_dim);
    t2 = init_tsmatrix(c->re->row_dim, c->re->col_dim);

    cmul_tsmatrix(t1, sc->val_re, a->re);
    cmul_tsmatrix(t2, sc->val_im, a->im);
    sub_tsmatrix(c->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        cmul_tsmatrix(t1, sc->val_im, a->re);
        cmul_tsmatrix(t2, sc->val_re, a->im);
        add_tsmatrix(c->im, t1, t2);
    //#else // USE_4M
        // 3M
    /*
        rts_add(tmp.val, sc->val_re, sc->val_im);
        t3 = init_tsmatrix(c->re->row_dim, c->im->col_dim);
        add_tsmatrix(t3, a->re, a->im);
        cmul_tsmatrix(c->im, tmp.val, t3);
        sub_tsmatrix(c->im, c->im, t1);
        sub_tsmatrix(c->im, c->im, t2);
        free_tsmatrix(t3);
    */
    //#endif // USE_4M

    free_tsmatrix(t1);
    free_tsmatrix(t2);
}

/* c := sc * a */
void cmul_ctsmatrix_3m(CTSMatrix c, ctsfloat *sc, CTSMatrix a)
{
    TSMatrix t1, t2, t3;
    tsfloat tmp;

    t1 = init_tsmatrix(c->re->row_dim, c->re->col_dim);
    t2 = init_tsmatrix(c->re->row_dim, c->re->col_dim);

    cmul_tsmatrix(t1, sc->val_re, a->re);
    cmul_tsmatrix(t2, sc->val_im, a->im);
    sub_tsmatrix(c->re, t1, t2);

    //#ifdef USE_4M
        // 4M
    /*
        cmul_tsmatrix(t1, sc->val_im, a->re);
        cmul_tsmatrix(t2, sc->val_re, a->im);
        add_tsmatrix(c->im, t1, t2);
    */
    //#else // USE_4M
        // 3M
        rts_add(tmp.val, sc->val_re, sc->val_im);
        t3 = init_tsmatrix(c->re->row_dim, c->im->col_dim);
        add_tsmatrix(t3, a->re, a->im);
        cmul_tsmatrix(c->im, tmp.val, t3);
        sub_tsmatrix(c->im, c->im, t1);
        sub_tsmatrix(c->im, c->im, t2);
        free_tsmatrix(t3);
    //#endif // USE_4M

    free_tsmatrix(t1);
    free_tsmatrix(t2);
}

/* c = a^T */
void transpose_ctsmatrix(CTSMatrix c, CTSMatrix a)
{
    transpose_tsmatrix(c->re, a->re);
    transpose_tsmatrix(c->im, a->im);
}

/* c := conj(a)^T */
void star_ctsmatrix(CTSMatrix c, CTSMatrix a)
{
    // c_re := -a_im
    _cts_neg_tsmatrix(c->re, a->im);
    // c_im := -a_im^T
    transpose_tsmatrix(c->im, c->re);
    // c_re := a_re^T
    transpose_tsmatrix(c->re, a->re);
}

/* c := conj(a) */
void conj_ctsmatrix(CTSMatrix c, CTSMatrix a)
{
    subst_tsmatrix(c->re, a->re);
    _cts_neg_tsmatrix(c->im, a->im);
}

/* c := -a */
void neg_ctsmatrix(CTSMatrix c, CTSMatrix a)
{
    _cts_neg_tsmatrix(c->re, a->re);
    _cts_neg_tsmatrix(c->im, a->im);
}

/* c := a */
void subst_ctsmatrix(CTSMatrix c, CTSMatrix a)
{
    subst_tsmatrix(c->re, a->re);
    subst_tsmatrix(c->im, a->im);
}

/* c := I */
void setI_ctsmatrix(CTSMatrix c)
{
    setI_tsmatrix(c->re);
    set0_tsmatrix(c->im);
}

/* v := a * vb */
void mul_ctsmatrix_ctsvec_4m(CTSVector v, CTSMatrix a, CTSVector vb)
{
    TSVector t1, t2, t3, t4;
    TSMatrix tmp_mat;

    t1 = init_tsvector(v->re->dim);
    t2 = init_tsvector(v->re->dim);
    t3 = init_tsvector(v->re->dim);
    t4 = init_tsvector(v->re->dim);

    mul_tsmatrix_tsvec(t1, a->re, vb->re);
    mul_tsmatrix_tsvec(t2, a->im, vb->im);
    sub_tsvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        mul_tsmatrix_tsvec(t3, a->im, vb->re);
        mul_tsmatrix_tsvec(t4, a->re, vb->im);
        add_tsvector(v->im, t3, t4);
    //#else // USE_4M
        // 3M
    /*
        tmp_mat = init_tsmatrix(a->re->row_dim, a->re->col_dim);
        add_tsmatrix(tmp_mat, a->re, a->im);
        add_tsvector(t3, vb->re, vb->im);
        mul_tsmatrix_tsvec(t4, tmp_mat, t3);
        sub_tsvector(v->im, t4, t1);
        sub_tsvector(v->im, v->im, t2);
        free_tsmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_tsvector(t1);
    free_tsvector(t2);
    free_tsvector(t3);
    free_tsvector(t4);
}

/* v := a * vb */
void mul_ctsmatrix_ctsvec_3m(CTSVector v, CTSMatrix a, CTSVector vb)
{
    TSVector t1, t2, t3, t4;
    TSMatrix tmp_mat;

    t1 = init_tsvector(v->re->dim);
    t2 = init_tsvector(v->re->dim);
    t3 = init_tsvector(v->re->dim);
    t4 = init_tsvector(v->re->dim);

    mul_tsmatrix_tsvec(t1, a->re, vb->re);
    mul_tsmatrix_tsvec(t2, a->im, vb->im);
    sub_tsvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
    /*
        mul_tsmatrix_tsvec(t3, a->im, vb->re);
        mul_tsmatrix_tsvec(t4, a->re, vb->im);
        add_tsvector(v->im, t3, t4);
    */
    //#else // USE_4M
        // 3M
        tmp_mat = init_tsmatrix(a->re->row_dim, a->re->col_dim);
        add_tsmatrix(tmp_mat, a->re, a->im);
        add_tsvector(t3, vb->re, vb->im);
        mul_tsmatrix_tsvec(t4, tmp_mat, t3);
        sub_tsvector(v->im, t4, t1);
        sub_tsvector(v->im, v->im, t2);
        free_tsmatrix(tmp_mat);
    //#endif // USE_4M

    free_tsvector(t1);
    free_tsvector(t2);
    free_tsvector(t3);
    free_tsvector(t4);
}


/* v := a^T * vb */
void mul_ctsmatrixt_ctsvec(CTSVector v, CTSMatrix a, CTSVector vb)
{
    TSVector t1, t2, t3, t4;
    TSMatrix tmp_mat;

    t1 = init_tsvector(v->re->dim);
    t2 = init_tsvector(v->re->dim);
    t3 = init_tsvector(v->re->dim);
    t4 = init_tsvector(v->re->dim);

    mul_tsmatrixt_tsvec(t1, a->re, vb->re);
    mul_tsmatrixt_tsvec(t2, a->im, vb->im);
    sub_tsvector(v->re, t1, t2);

   // #ifdef USE_4M
        // 4M
        mul_tsmatrixt_tsvec(t3, a->im, vb->re);
        mul_tsmatrixt_tsvec(t4, a->re, vb->im);
        add_tsvector(v->im, t3, t4);
   // #else // USE_4M
        // 3M
        /*
        tmp_mat = init_tsmatrix(a->re->row_dim, a->re->col_dim);
        add_tsmatrix(tmp_mat, a->re, a->im);
        add_tsvector(t3, vb->re, vb->im);
        mul_tsmatrixt_tsvec(t4, tmp_mat, t3);
        sub_tsvector(v->im, t4, t1);
        sub_tsvector(v->im, v->im, t2);
        free_tsmatrix(tmp_mat);
        */
    //#endif // USE_4M

    free_tsvector(t1);
    free_tsvector(t2);
    free_tsvector(t3);
    free_tsvector(t4);
}

/* v := conj(a)^T * vb */
void mul_ctsmatrixs_ctsvec(CTSVector v, CTSMatrix a, CTSVector vb)
{
    TSVector t1, t2, t3, t4;
    TSMatrix tmp_mat;

    t1 = init_tsvector(v->re->dim);
    t2 = init_tsvector(v->re->dim);
    t3 = init_tsvector(v->re->dim);
    t4 = init_tsvector(v->re->dim);

    mul_tsmatrixt_tsvec(t1, a->re, vb->re);
    mul_tsmatrixt_tsvec(t2, a->im, vb->im);
    add_tsvector(v->re, t1, t2);

   // #ifdef USE_4M
        // 4M
        mul_tsmatrixt_tsvec(t3, a->im, vb->re);
        mul_tsmatrixt_tsvec(t4, a->re, vb->im);
        sub_tsvector(v->im, t3, t4);
   // #else // USE_4M
        // 3M
        /*
        tmp_mat = init_tsmatrix(a->re->row_dim, a->re->col_dim);
        add_tsmatrix(tmp_mat, a->re, a->im);
        add_tsvector(t3, vb->re, vb->im);
        mul_tsmatrixt_tsvec(t4, tmp_mat, t3);
        sub_tsvector(v->im, t4, t1);
        sub_tsvector(v->im, v->im, t2);
        free_tsmatrix(tmp_mat);
        */
    //#endif // USE_4M

    free_tsvector(t1);
    free_tsvector(t2);
    free_tsvector(t3);
    free_tsvector(t4);
}


/* a = a^(-1) */
/* square matrix only */
void inv_ctsmatrix(CTSMatrix a)
{
	long int i, j, k, dim;
	ctsfloat ctmp, aii, aij, aik, ajk, aji;
	tsfloat tmp;

	/* Check Dimensions */
	if(a->re->row_dim != a->re->col_dim)
	{
		fprintf(stderr, "ERROR: inv_ctsmatrix\n");
		return;
	}

	dim = a->re->row_dim;

	for(i = 0; i < dim; i++)
	{
        subst_ctsmatrix_ij(&ctmp, a, i, i);
		rcts_abs(&tmp, &ctmp); // get_ctsmatrix_ij(a, i, i));
		if(rts_cmp_ui(tmp.val, 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_ctsmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		rcts_inv(&aii, &ctmp); // get_ctsmatrix_ij(a, i, i));
		//inv_mpc_t(aii, get_cmpfmatrix_ij(a, i, i));
		set_ctsmatrix_ij(a, i, i, &aii);

		for(j = 0; j < i; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
            subst_ctsmatrix_ij(&aij, a, i, j);
			rcts_mul(&ctmp, &aij, &aii); // get_ctsmatrix_ij(a, i, j), &aii);
			set_ctsmatrix_ij(a, i, j, &ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
            subst_ctsmatrix_ij(&aij, a, i, j);
			rcts_mul(&ctmp, &aij, &aii); // get_ctsmatrix_ij(a, i, j), &aii);
			set_ctsmatrix_ij(a, i, j, &ctmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_ctsmatrix_ij(&aji, a, j, i);
                subst_ctsmatrix_ij(&aik, a, i, k);
                subst_ctsmatrix_ij(&ajk, a, j, k);
				rcts_mul(&ctmp, &aji, &aik); // get_ctsmatrix_ij(a, j, i), get_ctsmatrix_ij(a, i, k));
				rcts_sub(&ctmp, &ajk, &ctmp); // get_ctsmatrix_ij(a, j, k), &ctmp);
				set_ctsmatrix_ij(a, j, k, &ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_ctsmatrix_ij(&aji, a, j, i);
                subst_ctsmatrix_ij(&aik, a, i, k);
                subst_ctsmatrix_ij(&ajk, a, j, k);
				rcts_mul(&ctmp, &aji, &aik); // get_ctsmatrix_ij(a, j, i), get_ctsmatrix_ij(a, i, k));
				rcts_sub(&ctmp, &ajk, &ctmp); // get_ctsmatrix_ij(a, j, k), &ctmp);
				set_ctsmatrix_ij(a, j, k, &ctmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_ctsmatrix_ij(&aji, a, j, i);
                subst_ctsmatrix_ij(&aik, a, i, k);
                subst_ctsmatrix_ij(&ajk, a, j, k);
				rcts_mul(&ctmp, &aji, &aik); // get_ctsmatrix_ij(a, j, i), get_ctsmatrix_ij(a, i, k));
				rcts_sub(&ctmp, &ajk, &ctmp); // get_ctsmatrix_ij(a, j, k), &ctmp);
				set_ctsmatrix_ij(a, j, k, &ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_ctsmatrix_ij(&aji, a, j, i);
                subst_ctsmatrix_ij(&aik, a, i, k);
                subst_ctsmatrix_ij(&ajk, a, j, k);
				rcts_mul(&ctmp, &aji, &aik); // get_ctsmatrix_ij(a, j, i), get_ctsmatrix_ij(a, i, k));
				rcts_sub(&ctmp, &ajk, &ctmp); // get_ctsmatrix_ij(a, j, k), &ctmp);
				set_ctsmatrix_ij(a, j, k, &ctmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			rcts_neg(&ctmp, &aii);
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
            subst_ctsmatrix_ij(&aji, a, j, i);
			rcts_mul(&ctmp, &ctmp, &aji); // get_ctsmatrix_ij(a, j, i));
			set_ctsmatrix_ij(a, j, i, &ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			rcts_neg(&ctmp, &aii);
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
            subst_ctsmatrix_ij(&aji, a, j, i);
            rcts_mul(&ctmp, &ctmp, &aji); // get_ctsmatrix_ij(a, j, i));
			set_ctsmatrix_ij(a, j, i, &ctmp);
		}
	}
}

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpc)a */
void subst_cmpfvector_ctsvec(CMPFVector c, CTSVector a)
{
    MPFVector re, im;

    re = init2_mpfvector(c->dim, c->prec);
    im = init2_mpfvector(c->dim, c->prec);

    subst_mpfvector_tsvec(re, a->re);
    subst_mpfvector_tsvec(im, a->im);

    merge_cmpfvector(c, re, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (ctd)a */
void subst_ctsvector_cmpfvec(CTSVector c, CMPFVector a)
{
    MPFVector re, im;

    re = init2_mpfvector(a->dim, a->prec);
    im = init2_mpfvector(a->dim, a->prec);

    separate_cmpfvector(re, im, a);
    subst_tsvector_mpfvec(c->re, re);
    subst_tsvector_mpfvec(c->im, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (mpf)a */
void subst_cmpfmatrix_ctsmat(CMPFMatrix c, CTSMatrix a)
{
    MPFMatrix re, im;

    re = init2_mpfmatrix(c->row_dim, c->col_dim, c->prec);
    im = init2_mpfmatrix(c->row_dim, c->col_dim, c->prec);

    subst_mpfmatrix_tsmat(re, a->re);
    subst_mpfmatrix_tsmat(im, a->im);

    merge_cmpfmatrix(c, re, im);

    free_mpfmatrix(re);
    free_mpfmatrix(im);
}


/* c := (ctd)a */
void subst_ctsmatrix_cmpfmat(CTSMatrix c, CMPFMatrix a)
{
    MPFMatrix re, im;

    re = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    im = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);

    separate_cmpfmatrix(re, im, a);
    subst_tsmatrix_mpfmat(c->re, re);
    subst_tsmatrix_mpfmat(c->im, im);

    free_mpfmatrix(re);
    free_mpfmatrix(im);
}

/* Normwise relative error of vector */
void relerr_ctsvector_cmpfvec(float relerr[TSSIZE], CTSVector approx_vec, CMPFVector true_vec, int norm_type)
{}

/* Elementwise relative errors of vector */
void relerr_element_ctsvector_mpf(float max_relerr[TSSIZE], float min_relerr[TSSIZE], float norm_relerr[TSSIZE], CTSVector approx_vec, MPFVector true_vec, int norm_type)
{}
#endif // USE_GMP

/* c := (td)a */
/* Normwise relative error of vector */
void relerr_ctsvector(float relerr[TSSIZE], CTSVector approx_vec, CTSVector true_vec, int norm_type)
{
    float norm_true_vec[TSSIZE], norm_diff_vec[TSSIZE];
	CTSVector diff_vec;

	diff_vec = init_ctsvector(approx_vec->re->dim);

	// diff_vec := approx_vec - true_vec
	sub_ctsvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_ctsvector(norm_diff_vec, diff_vec);
			normi_ctsvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_ctsvector(norm_diff_vec, diff_vec);
			norm1_ctsvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_ctsvector(norm_diff_vec, diff_vec);
			norm2_ctsvector(norm_true_vec, true_vec);
			break;
	}

	if(rts_cmp_ui(norm_true_vec, 0UL) != 0)
		rts_div(relerr, norm_diff_vec, norm_true_vec);

	free_ctsvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_ctsvector(float max_relerr[TSSIZE], float min_relerr[TSSIZE], float norm_relerr[TSSIZE], CTSVector approx_vec, CTSVector true_vec, int norm_type)
{
    tsfloat abs_true_vec, abs_diff_vec;
    float norm_diff_vec[TSSIZE], norm_true_vec[TSSIZE];
	long int i;
	CTSVector diff_vec;
    ctsfloat ctmp;

	diff_vec = init_ctsvector(approx_vec->re->dim);

	// diff_vec := approx_vec - true_vec
	sub_ctsvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_ctsvector(norm_diff_vec, diff_vec);
			normi_ctsvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_ctsvector(norm_diff_vec, diff_vec);
			norm1_ctsvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_ctsvector(norm_diff_vec, diff_vec);
			norm2_ctsvector(norm_true_vec, true_vec);
			break;
	}

	rts_set(norm_relerr, norm_diff_vec);
	if(rts_cmp_ui(norm_true_vec, 0UL) != 0)
		rts_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rts_set_ui(max_relerr, 0UL);
	normi_ctsvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->re->dim; i++)
	{
        subst_ctsvector_i(&ctmp, diff_vec, i);
		rcts_abs(&abs_diff_vec, &ctmp); // get_ctsvector_i(diff_vec, i));
        subst_ctsvector_i(&ctmp, true_vec, i);
		rcts_abs(&abs_true_vec, &ctmp); // get_ctsvector_i(true_vec, i));
		if(rts_cmp_ui(abs_true_vec.val, 0UL) != 0)
			rts_div(abs_diff_vec.val, abs_diff_vec.val, abs_true_vec.val);
		
		if(rts_cmp(max_relerr, abs_diff_vec.val) < 0)
			rts_set(max_relerr, abs_diff_vec.val);
		if(rts_cmp(min_relerr, abs_diff_vec.val) > 0)
			rts_set(min_relerr, abs_diff_vec.val);
	}

	free_ctsvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_ctsmatrix(CTSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
    row_swap_tsmatrix(mat->re, row_index0, row_index1, col_start, col_end);
    row_swap_tsmatrix(mat->im, row_index0, row_index1, col_start, col_end);
}

// print ctsmatrix
void print_ctsmatrix(CTSMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->re->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->re->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_dd_write((vec->element + index * TSSIZE));
			//c_dd_write(GET_DDMATRIX_IJ(mat, row_index, col_index));
			rts_out_str_base(stdout, 10, 48, get_ctsmatrix_ij_ctsfloat(mat, row_index, col_index).val_re);
            printf(" + ");
   			rts_out_str_base(stdout, 10, 48, get_ctsmatrix_ij_ctsfloat(mat, row_index, col_index).val_im);
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
    tsfloat a, b, c
    ctsfloat ca, cb, cc;

    rts_set_d(a->val, 3.0); rts_sqrt(a->val);
    rts_set_d(b->val, 5.0); rts_sqrt(b->val);

    rts_out_str_base(stdout, 10, 48, a);
    rts_out_str_base(stdout, 10, 48, b);
    
}
#endif // DEBUG

