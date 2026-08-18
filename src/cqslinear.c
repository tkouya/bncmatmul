/********************************************************************************/
/* cqdlinear.c: Quadruple-float precision Complex Linear Computation Library      */
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
#include "cqslinear.h"
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

// file-local: real qs matrix negation (no neg_qsmatrix in single-precision lib)
static void _cqs_neg_qsmatrix(QSMatrix c, QSMatrix a)
{
    float neg_one[QSSIZE];
    int i;
    neg_one[0] = -1.0f; for(i = 1; i < QSSIZE; i++) neg_one[i] = 0.0f;
    cmul_qsmatrix(c, neg_one, a);
}

// initialize CQSVector
CQSVector init_cqsvector(int dimension)
{
    CQSVector ret = NULL;

    ret = (CQSVector)malloc(sizeof(cqsvector));

    if(ret == NULL)
    {
        fprintf(stderr, "ERROR: init_cqsvector(%d)\n", dimension);
        return NULL;
    }

    ret->re = init_qsvector(dimension);
    if(ret->re == NULL)
    {
        fprintf(stderr, "ERROR: init_cqsvector(%d)\n", dimension);
        free(ret);
        return NULL;
    }

    ret->im = init_qsvector(dimension);
    if(ret->im == NULL)
    {
        fprintf(stderr, "ERROR: init_cqsvector(%d)\n", dimension);
        free(ret->re);
        free(ret);
        return NULL;
    }

    return ret;
}

// free CQSVector
void free_cqsvector(CQSVector vec)
{
    free_qsvector(vec->re);
    free_qsvector(vec->im);
    free(vec);
}

// CQSVector vec -> qsfloat array
void set_cqsfloat_cqsvec(cqsfloat ret[], int ret_dim, CQSVector vec)
{
    int i;
    cqsfloat ptr_val;

    for(i = 0; i < ret_dim; i++)
        subst_cqsvector_i(&ret[i], vec, i);
        //rcqs_set(&ret[i], get_cqsvector_i(vec, i));

/*    {
        ptr_val = get_cqsvector_i_cqsfloat(vec, i);
        rcqs_set(&ret[i], &ptr_val);
    }
*/

    return;
}

// cqsfloat array -> CQSVector ret
void set_cqsvector_cqsfloat(CQSVector ret, cqsfloat array[], int array_dim)
{
    int i;

    for(i = 0; i < ret->re->dim; i++)
        set_cqsvector_i(ret, i, &array[i]);
}

// qsvector -> cqsvector
void set_cqsvector_qsvec(CQSVector ret, QSVector re_vec, QSVector im_vec)
{
    subst_qsvector(ret->re, re_vec);
    subst_qsvector(ret->im, im_vec);
}

// print qsvector
void print_cqsvector(CQSVector vec)
{
    int i;

    for(i = 0; i < vec->re->dim; i++)
    {
        printf("%5d ", i);
        rqs_out_str_base(stdout, 10, 64, get_qsvector_i(vec->re, i));
        printf(" + ");
        rqs_out_str_base(stdout, 10, 64, get_qsvector_i(vec->im, i));
        printf("\n");
    }
}

// set a zero vector
void set0_cqsvector(CQSVector vec)
{
    set0_qsvector(vec->re);
    set0_qsvector(vec->im);
}

/*************************************************/
/* Vector Calculations for CQSVector               */
/*
void add_cqsvector(CQSVector c, CQSVector a, CQSVector b)
void add2_cqsvector(CQSVector c, CQSVector a)
void sub_cqsvector(CQSVector c, CQSVector a, CQSVector b)
void sub2_cqsvector(CQSVector c, DVector a)
void cmul_cqsvector(CQSVector c, float val[QSSIZE], CQSVector a)
void cmul2_cqsvector(CQSVector c, float val[QSSIZE])
void add_cmul_cqsvector(CQSVector c, CQSVector a, float val[QSSIZE], CQSVector b)
float ip_cqsvector(CQSVector a, CQSVector b)
float norm1_cqsvector(CQSVector a)
float norm2_cqsvector(CQSVector a)
float normi_cqsvector(CQSVector a)
void subst_cqsvector(CQSVector c, CQSVector a)
*/
/*************************************************/
/* c = a + b */
void add_cqsvector(CQSVector c, CQSVector a, CQSVector b)
{
    add_qsvector(c->re, a->re, b->re);
    add_qsvector(c->im, a->im, b->im);
}

/* c += a */
void add2_cqsvector(CQSVector c, CQSVector a)
{
    add2_qsvector(c->re, a->re);
    add2_qsvector(c->im, a->im);
}

/* c = a - b */
void sub_cqsvector(CQSVector c, CQSVector a, CQSVector b)
{
    sub_qsvector(c->re, a->re, b->re);
    sub_qsvector(c->im, a->im, b->im);
}

/* c -= a */
void sub2_cqsvector(CQSVector c, CQSVector a)
{
    sub2_qsvector(c->re, a->re);
    sub2_qsvector(c->im, a->im);
}

/* c = val * a */
void cmul_cqsvector_4m(CQSVector c, cqsfloat *val, CQSVector a)
{
    QSVector t1, t2, t3, t4;

    // 2024-11-28(Thu) Fixed! T.Kouya
    t1 = init_qsvector(c->re->dim);
    t2 = init_qsvector(c->re->dim);
    t3 = init_qsvector(c->re->dim);
    t4 = init_qsvector(c->re->dim);

    cmul_qsvector(t1, val->val_re, a->re);
    cmul_qsvector(t2, val->val_im, a->im);  
    cmul_qsvector(t3, val->val_im, a->re);
    cmul_qsvector(t4, val->val_re, a->im);

    sub_qsvector(c->re, t1, t2);
    add_qsvector(c->im, t3, t4);

    free_qsvector(t1);
    free_qsvector(t2);
    free_qsvector(t3);
    free_qsvector(t4);

}

/* c = val * a */
void cmul_cqsvector_3m(CQSVector c, cqsfloat *val, CQSVector a)
{
    QSVector t1, t2, t3;
    qsfloat tmp;

    t1 = init_qsvector(c->re->dim);
    t2 = init_qsvector(c->re->dim);

    cmul_qsvector(t1, val->val_re, a->re);
    cmul_qsvector(t2, val->val_im, a->im);
    sub_qsvector(c->re, t1, t2);

    //#ifdef USE_4M
        // 4M
    /*
        cmul_qsvector(t1, val->val_im, a->re);
        cmul_qsvector(t2, val->val_re, a->im);
        add_qsvector(c->im, t1, t2);
    */
    //#else // USE_4M
        // 3M
        rqs_add(tmp.val, val->val_re, val->val_im);
        t3 = init_qsvector(c->re->dim);
        add_qsvector(t3, a->re, a->im);
        cmul_qsvector(c->im, tmp.val, t3);
        sub_qsvector(c->im, c->im, t1);
        sub_qsvector(c->im, c->im, t2);
        free_qsvector(t3);
    //#endif // USE_4M

    free_qsvector(t1);
    free_qsvector(t2);
}


/* c *= val */
void cmul2_cqsvector(CQSVector c, cqsfloat *val)
{
    CQSVector in_a;

    in_a = init_cqsvector(c->re->dim);

    subst_cqsvector(in_a, c);
    cmul_cqsvector(c, val, in_a);

    free_cqsvector(in_a);
}

/* c = a + val * b */
void add_cmul_cqsvector(CQSVector c, CQSVector a, cqsfloat *val, CQSVector b)
{
    CQSVector in_vec;
    in_vec = init_cqsvector(b->re->dim);

    //cmul_cqsvector(c, val, b);
    cmul_cqsvector(in_vec, val, b);
    //add2_cqsvector(c, a);
    add_cqsvector(c, a, in_vec);

    free_cqsvector(in_vec);
}

/* c = a - val * b */
void sub_cmul_cqsvector(CQSVector c, CQSVector a, cqsfloat *val, CQSVector b)
{
    CQSVector in_vec;
    in_vec = init_cqsvector(b->re->dim);

    //cmul_cqsvector(c, val, b);
    cmul_cqsvector(in_vec, val, b);
    //add2_cqsvector(c, a);
    sub_cqsvector(c, a, in_vec);

    free_cqsvector(in_vec);
}

/* (a, b) = conj(a)^T * b */
void ip_cqsvector(cqsfloat *ret, CQSVector a, CQSVector b)
{
    int i;
    cqsfloat tmp, conj_a_i, b_i;

    rcqs_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cqsvector_i(&tmp, a, i);
        rcqs_conj(&conj_a_i, &tmp); //get_cqsvector_i(a, i));
        subst_cqsvector_i(&b_i, b, i);
        rcqs_mul(&tmp, &conj_a_i, &b_i); //get_cqsvector_i(b, i));
        rcqs_add(ret, ret, &tmp);
    }
}

/* a^T * b */
void dotp_cqsvector(cqsfloat *ret, CQSVector a, CQSVector b)
{
    int i;
    cqsfloat tmp, ai, bi;

    rcqs_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cqsvector_i(&ai, a, i);
        subst_cqsvector_i(&bi, b, i);
        rcqs_mul(&tmp, &ai, &bi); //get_cqsvector_i(a, i), get_cqsvector_i(b, i));
        rcqs_add(ret, ret, &tmp);
    }
}

/* c := a */
void subst_cqsvector(CQSVector c, CQSVector a)
{
    subst_qsvector(c->re, a->re);
    subst_qsvector(c->im, a->im);
}

/* c := conj(a) */
void conj_cqsvector(CQSVector c, CQSVector a)
{
    subst_qsvector(c->re, a->re);
    neg_qsvector(c->im, a->im);
}

/* c := -a */
void neg_cqsvector(CQSVector c, CQSVector a)
{
    neg_qsvector(c->re, a->re);
    neg_qsvector(c->im, a->im);
}

/* ||a||_1 */
void norm1_cqsvector(float ret[QSSIZE], CQSVector a)
{
    int i;
    qsfloat tmp;
    cqsfloat ctmp;

    rqs_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cqsvector_i(&ctmp, a, i);
        rcqs_abs(&tmp, &ctmp);
        rqs_add(ret, ret, tmp.val);
    }
}

/* ||a||_infty */
void normi_cqsvector(float ret[QSSIZE], CQSVector a)
{
    int i;
    qsfloat tmp;
    cqsfloat ctmp;

    rqs_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cqsvector_i(&ctmp, a, i);
        rcqs_abs(&tmp, &ctmp);
        if(rqs_cmp(ret, tmp.val) < 0)
            rqs_set(ret, tmp.val);
    }
}

// Euclid norm
void norm2_cqsvector(float ret[QSSIZE], CQSVector vec)
{
    int i;
    qsfloat tmp;
    cqsfloat ctmp;

    rqs_set0(ret);
    for(i = 0; i < vec->re->dim; i++)
    {
        subst_cqsvector_i(&ctmp, vec, i);
        rcqs_nrm2(&tmp, &ctmp);
        rqs_add(ret, ret, tmp.val);
    }
    #ifdef USE_GMP
    rqs_sqrt_mpfr(ret, ret);
    #else // USE_GMP
    rqs_sqrt(ret, ret);
    #endif // USE_GMP
}

// set a zero matrix
//void set0_cqsmatrix(CQSMatrix mat);
void set0_cqsmatrix(CQSMatrix mat)
{
    set0_qsmatrix(mat->re);
    set0_qsmatrix(mat->im);
}

// initialize qsvector
CQSMatrix init_cqsmatrix(long int row_dim, long int col_dim)
{
    CQSMatrix ret = NULL;

    ret = (CQSMatrix)malloc(sizeof(cqsmatrix));
    if(ret == NULL)
    {
        fprintf(stderr, "ERROR: init_cqsmatrix(%ld, %ld)\n", row_dim, col_dim);
        return NULL;
    }

    ret->re = init_qsmatrix(row_dim, col_dim);
    if(ret->re == NULL)
    {
        fprintf(stderr, "ERROR: init_cqsmatrix(%ld, %ld)\n", row_dim, col_dim);
        free(ret);
        return NULL;
    }

    ret->im = init_qsmatrix(row_dim, col_dim);
    if(ret->im == NULL)
    {
        fprintf(stderr, "ERROR: init_cqsmatrix(%ld, %ld)\n", row_dim, col_dim);
        free_qsmatrix(ret->re);
        free(ret);
        return NULL;
    }

    return ret;
}

// free csqsvector
void free_cqsmatrix(CQSMatrix mat)
{
    free_qsmatrix(mat->re);
    free_qsmatrix(mat->im);
    free(mat);
}

// print cqsvector
//void print_cqsmatrix(CQSMatrix mat);

// qsmatrix -> cqsmatrix
void set_cqsmatrix_qsmat(CQSMatrix ret, QSMatrix re_mat, QSMatrix im_mat)
{
    subst_qsmatrix(ret->re, re_mat);
    subst_qsmatrix(ret->im, im_mat);
}

// CQSMatrix mat -> cqsfloat array
void set_cqsfloat_cqsmat(cqsfloat ret[], int ret_dim, CQSMatrix mat)
{
    long int i, j, index;
    cqsfloat mat_ij;

    index = 0;
    for(i = 0; i < mat->re->row_dim; i++)
    {
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_cqsmatrix_ij(&mat_ij, mat, i, j); 
            rcqs_set(&ret[index++], &mat_ij); //get_cqsmatrix_ij(mat, i, j));
        }

    }

    return;
}

// qsfloat array -> DDmatrix ret
void set_cqsmatrix_cqsfloat(CQSMatrix ret, cqsfloat array[], int array_dim)
{
    long int i, j, index;

    index = 0;
    for(i = 0; i < ret->re->row_dim; i++)
    {
        for(j = 0; j < ret->re->col_dim; j++)
            set_cqsmatrix_ij(ret, i, j, &array[index++]);
    }

    return;
}

// matrix multiplication
// ret := A * B
void mul_cqsmatrix_4m(CQSMatrix ret, CQSMatrix a, CQSMatrix b)
{
    QSMatrix t1, t2, t3, t4;

    t1 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_qsmatrix(t1, a->re, b->re);
    mul_qsmatrix(t2, a->im, b->im);
    sub_qsmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_qsmatrix(t3, a->im, b->re);
        mul_qsmatrix(t4, a->re, b->im);
        add_qsmatrix(ret->im, t3, t4);
    //#else // USE_4M 
        // 3M
    /*
        add_qsmatrix(t3, a->re, a->im);
        add_qsmatrix(t4, b->re, b->im);
        mul_qsmatrix(ret->im, t3, t4);
        sub_qsmatrix(ret->im, ret->im, t1);
        sub_qsmatrix(ret->im, ret->im, t2);
    */
    //#endif // USE_4M

    free_qsmatrix(t1);
    free_qsmatrix(t2);
    free_qsmatrix(t3);
    free_qsmatrix(t4);
}

// matrix multiplication
// ret := A * B
void mul_cqsmatrix_3m(CQSMatrix ret, CQSMatrix a, CQSMatrix b)
{
    QSMatrix t1, t2, t3, t4;

    t1 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qsmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_qsmatrix(b->re->row_dim, ret->re->col_dim);

    mul_qsmatrix(t1, a->re, b->re);
    mul_qsmatrix(t2, a->im, b->im);
    sub_qsmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
    /*
        mul_qsmatrix(t3, a->im, b->re);
        mul_qsmatrix(t4, a->re, b->im);
        add_qsmatrix(ret->im, t1, t2);
    */
    //#else // USE_4M 
        // 3M
        add_qsmatrix(t3, a->re, a->im);
        add_qsmatrix(t4, b->re, b->im);
        mul_qsmatrix(ret->im, t3, t4);
        sub_qsmatrix(ret->im, ret->im, t1);
        sub_qsmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_qsmatrix(t1);
    free_qsmatrix(t2);
    free_qsmatrix(t3);
    free_qsmatrix(t4);
}

// Frobenius norm
void normf_cqsmatrix(float ret[QSSIZE], CQSMatrix mat)
{
    int i, j;
    qsfloat tmp;
    cqsfloat ctmp;

    rqs_set0(ret);
    for(i = 0; i < mat->re->row_dim; i++)
    {
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_cqsmatrix_ij(&ctmp, mat, i, j);
            rcqs_nrm2(&tmp, &ctmp);
            rqs_add(ret, ret, tmp.val);
        }
    }
    rqs_sqrt(ret, ret);
}

// print normf
void print_normf_cqsmatrix(const char *str, CQSMatrix mat)
{
    qsfloat tmp;

    normf_cqsmatrix(tmp.val, mat);

    rqs_out_str_base(stdout, 10, 64, tmp.val);
}

/*************************************************/
/* Matrix Caluculations for CQSMatrix            */
/*
void normf_cqsmatrix(float ret[QSSIZE], CQSMatrix mat)
void norm1_cqsmatrix(float ret[QSSIZE], CQSMatrix mat)
void normi_cqsmatrix(float ret[QSSIZE], CQSMatrix mat)
void add_cqsmatrix(CQSMatrix c, CQSMatrix a, CQSMatrix b);
void sub_cqsmatrix(CQSMatrix c, CQSMatrix a, CQSMatrix b);
void mul_cqsmatrix(CQSMatrix c, CQSMatrix a, CQSMatrix b);
void mul_cqsmatrix_cqsvec(CQSVector v, CQSMatrix a, CQSVector vb)
void mul_cqsmatrixt_cqsvec(CQSVector v, CQSMatrix a, CQSVector vb)
void transpose_cqsmatrix(CQSMatrix c, CQSMatrix a);
void inv_cqsmatrix(CQSMatrix a);
void subst_mpfmatrux(CQSMatrix c, CQSMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_cqsmatrix(float ret[QSSIZE], CQSMatrix mat)
{
    int i, j;
    qsfloat tmp, sum;
    cqsfloat ctmp;

    rqs_set0(ret);
    for(j = 0; j < mat->re->col_dim; j++)
    {
        rqs_set0(sum.val);
        for(i = 0; i < mat->re->row_dim; i++)
        {
            subst_cqsmatrix_ij(&ctmp, mat, i, j);
            rcqs_abs(&tmp, &ctmp);
            rqs_add(sum.val, sum.val, tmp.val);
        }
        if(rqs_cmp(ret, sum.val) < 0)
            rqs_set(ret, sum.val);
    }
}

/* 1 Norm of Matrix */
void norm1_cqsmatrix(float ret[QSSIZE], CQSMatrix mat)
{
    int i, j;
    qsfloat tmp, sum;
    cqsfloat ctmp;

    rqs_set0(ret);
    for(i = 0; i < mat->re->row_dim; i++)
    {
        rqs_set0(sum.val);
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_cqsmatrix_ij(&ctmp, mat, i, j);
            rcqs_abs(&tmp, &ctmp);
            rqs_add(sum.val, sum.val, tmp.val);
        }
        if(rqs_cmp(ret, sum.val) < 0)
            rqs_set(ret, sum.val);
    }
}

/* c := a + b */
void add_cqsmatrix(CQSMatrix c, CQSMatrix a, CQSMatrix b)
{
    add_qsmatrix(c->re, a->re, b->re);
    add_qsmatrix(c->im, a->im, b->im);
}

/* c := a - b */
void sub_cqsmatrix(CQSMatrix c, CQSMatrix a, CQSMatrix b)
{
    sub_qsmatrix(c->re, a->re, b->re);
    sub_qsmatrix(c->im, a->im, b->im);
}

/* c := sc * a */
void cmul_cqsmatrix(CQSMatrix c, cqsfloat *sc, CQSMatrix a)
{
    QSMatrix t1, t2, t3;
    qsfloat tmp;

    t1 = init_qsmatrix(c->re->row_dim, c->re->col_dim);
    t2 = init_qsmatrix(c->re->row_dim, c->re->col_dim);

    cmul_qsmatrix(t1, sc->val_re, a->re);
    cmul_qsmatrix(t2, sc->val_im, a->im);
    sub_qsmatrix(c->re, t1, t2);

    #ifdef USE_4M
        // 4M
        cmul_qsmatrix(t1, sc->val_im, a->re);
        cmul_qsmatrix(t2, sc->val_re, a->im);
        add_qsmatrix(c->im, t1, t2);
    #else // USE_4M
        // 3M
        rqs_add(tmp.val, sc->val_re, sc->val_im);
        t3 = init_qsmatrix(c->re->row_dim, c->im->col_dim);
        add_qsmatrix(t3, a->re, a->im);
        cmul_qsmatrix(c->im, tmp.val, t3);
        sub_qsmatrix(c->im, c->im, t1);
        sub_qsmatrix(c->im, c->im, t2);
        free_qsmatrix(t3);
    #endif // USE_4M

    free_qsmatrix(t1);
    free_qsmatrix(t2);
}

/* c = a^T */
void transpose_cqsmatrix(CQSMatrix c, CQSMatrix a)
{
    transpose_qsmatrix(c->re, a->re);
    transpose_qsmatrix(c->im, a->im);
}

/* c := conj(a)^T */
void star_cqsmatrix(CQSMatrix c, CQSMatrix a)
{
    // c_re := -a_im
    _cqs_neg_qsmatrix(c->re, a->im);
    // c_im := -a_im^T
    transpose_qsmatrix(c->im, c->re);
    // c_re := a_re^T
    transpose_qsmatrix(c->re, a->re);
}

/* c := conj(a) */
void conj_cqsmatrix(CQSMatrix c, CQSMatrix a)
{
    subst_qsmatrix(c->re, a->re);
    _cqs_neg_qsmatrix(c->im, a->im);
}

/* c := -a */
void neg_cqsmatrix(CQSMatrix c, CQSMatrix a)
{
    _cqs_neg_qsmatrix(c->re, a->re);
    _cqs_neg_qsmatrix(c->im, a->im);
}

/* c := a */
void subst_cqsmatrix(CQSMatrix c, CQSMatrix a)
{
    subst_qsmatrix(c->re, a->re);
    subst_qsmatrix(c->im, a->im);
}

/* c := I */
void setI_cqsmatrix(CQSMatrix c)
{
    setI_qsmatrix(c->re);
    set0_qsmatrix(c->im);
}

/* v := a * vb */
void mul_cqsmatrix_cqsvec_4m(CQSVector v, CQSMatrix a, CQSVector vb)
{
    QSVector t1, t2, t3, t4;
    QSMatrix tmp_mat;

    t1 = init_qsvector(v->re->dim);
    t2 = init_qsvector(v->re->dim);
    t3 = init_qsvector(v->re->dim);
    t4 = init_qsvector(v->re->dim);

    mul_qsmatrix_qsvec(t1, a->re, vb->re);
    mul_qsmatrix_qsvec(t2, a->im, vb->im);
    sub_qsvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        mul_qsmatrix_qsvec(t3, a->im, vb->re);
        mul_qsmatrix_qsvec(t4, a->re, vb->im);
        add_qsvector(v->im, t3, t4);
    //#else // USE_4M
    /*
        // 3M
        tmp_mat = init_qsmatrix(a->re->row_dim, a->re->col_dim);
        add_qsmatrix(tmp_mat, a->re, a->im);
        add_qsvector(t3, vb->re, vb->im);
        mul_qsmatrix_qsvec(t4, tmp_mat, t3);
        sub_qsvector(v->im, t4, t1);
        sub_qsvector(v->im, v->im, t2);
        free_qsmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_qsvector(t1);
    free_qsvector(t2);
    free_qsvector(t3);
    free_qsvector(t4);
}

/* v := a * vb */
void mul_cqsmatrix_cqsvec_3m(CQSVector v, CQSMatrix a, CQSVector vb)
{
    QSVector t1, t2, t3, t4;
    QSMatrix tmp_mat;

    t1 = init_qsvector(v->re->dim);
    t2 = init_qsvector(v->re->dim);
    t3 = init_qsvector(v->re->dim);
    t4 = init_qsvector(v->re->dim);

    mul_qsmatrix_qsvec(t1, a->re, vb->re);
    mul_qsmatrix_qsvec(t2, a->im, vb->im);
    sub_qsvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
    /*
        mul_qsmatrix_qsvec(t3, a->im, vb->re);
        mul_qsmatrix_qsvec(t4, a->re, vb->im);
        add_qsvector(v->im, t3, t4);
    */
    //#else // USE_4M
        // 3M
        tmp_mat = init_qsmatrix(a->re->row_dim, a->re->col_dim);
        add_qsmatrix(tmp_mat, a->re, a->im);
        add_qsvector(t3, vb->re, vb->im);
        mul_qsmatrix_qsvec(t4, tmp_mat, t3);
        sub_qsvector(v->im, t4, t1);
        sub_qsvector(v->im, v->im, t2);
        free_qsmatrix(tmp_mat);
    //#endif // USE_4M

    free_qsvector(t1);
    free_qsvector(t2);
    free_qsvector(t3);
    free_qsvector(t4);
}

/* v := a^T * vb */
void mul_cqsmatrixt_cqsvec(CQSVector v, CQSMatrix a, CQSVector vb)
{
    QSVector t1, t2, t3, t4;
    QSMatrix tmp_mat;

    t1 = init_qsvector(v->re->dim);
    t2 = init_qsvector(v->re->dim);
    t3 = init_qsvector(v->re->dim);
    t4 = init_qsvector(v->re->dim);

    mul_qsmatrixt_qsvec(t1, a->re, vb->re);
    mul_qsmatrixt_qsvec(t2, a->im, vb->im);
    sub_qsvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        mul_qsmatrixt_qsvec(t3, a->im, vb->re);
        mul_qsmatrixt_qsvec(t4, a->re, vb->im);
        add_qsvector(v->im, t3, t4);
    //#else // USE_4M
        // 3M
    /*
        tmp_mat = init_qsmatrix(a->re->row_dim, a->re->col_dim);
        add_qsmatrix(tmp_mat, a->re, a->im);
        add_qsvector(t3, vb->re, vb->im);
        mul_qsmatrixt_qsvec(t4, tmp_mat, t3);
        sub_qsvector(v->im, t4, t1);
        sub_qsvector(v->im, v->im, t2);
        free_qsmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_qsvector(t1);
    free_qsvector(t2);
    free_qsvector(t3);
    free_qsvector(t4);
}

/* v := conj(a)^T * vb */
void mul_cqsmatrixs_cqsvec(CQSVector v, CQSMatrix a, CQSVector vb)
{
    QSVector t1, t2, t3, t4;
    QSMatrix tmp_mat;

    t1 = init_qsvector(v->re->dim);
    t2 = init_qsvector(v->re->dim);
    t3 = init_qsvector(v->re->dim);
    t4 = init_qsvector(v->re->dim);

    mul_qsmatrixt_qsvec(t1, a->re, vb->re);
    mul_qsmatrixt_qsvec(t2, a->im, vb->im);
    add_qsvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        mul_qsmatrixt_qsvec(t3, a->im, vb->re);
        mul_qsmatrixt_qsvec(t4, a->re, vb->im);
        sub_qsvector(v->im, t3, t4);
    //#else // USE_4M
        // 3M
    /*
        tmp_mat = init_qsmatrix(a->re->row_dim, a->re->col_dim);
        add_qsmatrix(tmp_mat, a->re, a->im);
        add_qsvector(t3, vb->re, vb->im);
        mul_qsmatrixt_qsvec(t4, tmp_mat, t3);
        sub_qsvector(v->im, t4, t1);
        sub_qsvector(v->im, v->im, t2);
        free_qsmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_qsvector(t1);
    free_qsvector(t2);
    free_qsvector(t3);
    free_qsvector(t4);
}

/* a = a^(-1) */
/* square matrix only */
void inv_cqsmatrix(CQSMatrix a)
{
	long int i, j, k, dim;
	cqsfloat ctmp, aii, aij, aik, ajk, aji;
	qsfloat tmp;

	/* Check Dimensions */
	if(a->re->row_dim != a->re->col_dim)
	{
		fprintf(stderr, "ERROR: inv_cqsmatrix\n");
		return;
	}

	dim = a->re->row_dim;

	for(i = 0; i < dim; i++)
	{
        subst_cqsmatrix_ij(&aii, a, i, i);
		rcqs_abs(&tmp, &aii); //get_cqsmatrix_ij(a, i, i));
		if(rqs_cmp_ui(tmp.val, 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_cqsmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		rcqs_inv(&aii, &aii); //get_cqsmatrix_ij(a, i, i));
		//inv_mpc_t(aii, get_cmpfmatrix_ij(a, i, i));
		set_cqsmatrix_ij(a, i, i, &aii);

		for(j = 0; j < i; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
            subst_cqsmatrix_ij(&aij, a, i, j);
			rcqs_mul(&ctmp, &aij, &aii); //get_cqsmatrix_ij(a, i, j), &aii);
			set_cqsmatrix_ij(a, i, j, &ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
            subst_cqsmatrix_ij(&aij, a, i, j);
			rcqs_mul(&ctmp, &aij, &aii); //get_cqsmatrix_ij(a, i, j), &aii);
			set_cqsmatrix_ij(a, i, j, &ctmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cqsmatrix_ij(&aji, a, j, i);
                subst_cqsmatrix_ij(&aik, a, i, k);
				subst_cqsmatrix_ij(&ajk, a, j, k);
				rcqs_mul(&ctmp, &aji, &aik); //get_cqsmatrix_ij(a, i, k));
				rcqs_sub(&ctmp, &ajk, &ctmp);
				set_cqsmatrix_ij(a, j, k, &ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cqsmatrix_ij(&aji, a, j, i);
                subst_cqsmatrix_ij(&aik, a, i, k);
				subst_cqsmatrix_ij(&ajk, a, j, k);
				rcqs_mul(&ctmp, &aji, &aik); //get_cqsmatrix_ij(a, j, i), get_cqsmatrix_ij(a, i, k));
				rcqs_sub(&ctmp, &ajk, &ctmp); //get_cqsmatrix_ij(a, j, k), &ctmp);
				set_cqsmatrix_ij(a, j, k, &ctmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cqsmatrix_ij(&aji, a, j, i);
                subst_cqsmatrix_ij(&aik, a, i, k);
				subst_cqsmatrix_ij(&ajk, a, j, k);
				rcqs_mul(&ctmp, &aji, &aik); //get_cqsmatrix_ij(a, j, i), get_cqsmatrix_ij(a, i, k));
				rcqs_sub(&ctmp, &ajk, &ctmp); //get_cqsmatrix_ij(a, j, k), &ctmp);
				set_cqsmatrix_ij(a, j, k, &ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cqsmatrix_ij(&aji, a, j, i);
                subst_cqsmatrix_ij(&aik, a, i, k);
				subst_cqsmatrix_ij(&ajk, a, j, k);
				rcqs_mul(&ctmp, &aji, &aik); //get_cqsmatrix_ij(a, j, i), get_cqsmatrix_ij(a, i, k));
				rcqs_sub(&ctmp, &ajk, &ctmp); //get_cqsmatrix_ij(a, j, k), &ctmp);
				set_cqsmatrix_ij(a, j, k, &ctmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			rcqs_neg(&ctmp, &aii);
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
            subst_cqsmatrix_ij(&aji, a, j, i);
			rcqs_mul(&ctmp, &ctmp, &aji); //get_cqsmatrix_ij(a, j, i));
			set_cqsmatrix_ij(a, j, i, &ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			rcqs_neg(&ctmp, &aii);
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
            subst_cqsmatrix_ij(&aji, a, j, i);
			rcqs_mul(&ctmp, &ctmp, &aji); //get_cqsmatrix_ij(a, j, i));
			set_cqsmatrix_ij(a, j, i, &ctmp);
		}
	}
}

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpc)a */
void subst_cmpfvector_cqsvec(CMPFVector c, CQSVector a)
{
    MPFVector re, im;

    re = init2_mpfvector(c->dim, c->prec);
    im = init2_mpfvector(c->dim, c->prec);

    subst_mpfvector_qsvec(re, a->re);
    subst_mpfvector_qsvec(im, a->im);

    merge_cmpfvector(c, re, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (dd)a */
void subst_cqsvector_cmpfvec(CQSVector c, CMPFVector a)
{
    MPFVector re, im;

    re = init2_mpfvector(a->dim, a->prec);
    im = init2_mpfvector(a->dim, a->prec);

    separate_cmpfvector(re, im, a);
    subst_qsvector_mpfvec(c->re, re);
    subst_qsvector_mpfvec(c->im, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (mpf)a */
void subst_cmpfmatrix_cqsmat(CMPFMatrix c, CQSMatrix a)
{
    MPFMatrix re, im;

    re = init2_mpfmatrix(c->row_dim, c->col_dim, c->prec);
    im = init2_mpfmatrix(c->row_dim, c->col_dim, c->prec);

    subst_mpfmatrix_qsmat(re, a->re);
    subst_mpfmatrix_qsmat(im, a->im);

    merge_cmpfmatrix(c, re, im);

    free_mpfmatrix(re);
    free_mpfmatrix(im);
}


/* c := (dd)a */
void subst_cqsmatrix_cmpfmat(CQSMatrix c, CMPFMatrix a)
{
    MPFMatrix re, im;

    re = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    im = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);

    separate_cmpfmatrix(re, im, a);
    subst_qsmatrix_mpfmat(c->re, re);
    subst_qsmatrix_mpfmat(c->im, im);

    free_mpfmatrix(re);
    free_mpfmatrix(im);
}

/* Normwise relative error of vector */
void relerr_cqsvector_cmpfvec(float relerr[QSSIZE], CQSVector approx_vec, CMPFVector true_vec, int norm_type)
{}

/* Elementwise relative errors of vector */
void relerr_element_cqsvector_mpf(float max_relerr[QSSIZE], float min_relerr[QSSIZE], float norm_relerr[QSSIZE], CQSVector approx_vec, MPFVector true_vec, int norm_type)
{}
#endif // USE_GMP

/* Normwise relative error of vector */
void relerr_cqsvector(float relerr[QSSIZE], CQSVector approx_vec, CQSVector true_vec, int norm_type)
{
    float norm_true_vec[QSSIZE], norm_diff_vec[QSSIZE];
	CQSVector diff_vec;

	diff_vec = init_cqsvector(approx_vec->re->dim);

	// diff_vec := approx_vec - true_vec
	sub_cqsvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_cqsvector(norm_diff_vec, diff_vec);
			normi_cqsvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_cqsvector(norm_diff_vec, diff_vec);
			norm1_cqsvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_cqsvector(norm_diff_vec, diff_vec);
			norm2_cqsvector(norm_true_vec, true_vec);
			break;
	}

	if(rqs_cmp_ui(norm_true_vec, 0UL) != 0)
		rqs_div(relerr, norm_diff_vec, norm_true_vec);

	free_cqsvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_cqsvector(float max_relerr[QSSIZE], float min_relerr[QSSIZE], float norm_relerr[QSSIZE], CQSVector approx_vec, CQSVector true_vec, int norm_type)
{
    qsfloat abs_true_vec, abs_diff_vec;
    float norm_diff_vec[QSSIZE], norm_true_vec[QSSIZE];
	long int i;
	CQSVector diff_vec;
    cqsfloat ctmp;

	diff_vec = init_cqsvector(approx_vec->re->dim);

	// diff_vec := approx_vec - true_vec
	sub_cqsvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_cqsvector(norm_diff_vec, diff_vec);
			normi_cqsvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_cqsvector(norm_diff_vec, diff_vec);
			norm1_cqsvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_cqsvector(norm_diff_vec, diff_vec);
			norm2_cqsvector(norm_true_vec, true_vec);
			break;
	}

	rqs_set(norm_relerr, norm_diff_vec);
	if(rqs_cmp_ui(norm_true_vec, 0UL) != 0)
		rqs_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rqs_set_ui(max_relerr, 0UL);
	normi_cqsvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->re->dim; i++)
	{
        subst_cqsvector_i(&ctmp, diff_vec, i);
		rcqs_abs(&abs_diff_vec, &ctmp); // get_cqsvector_i(diff_vec, i));
        subst_cqsvector_i(&ctmp, true_vec, i);
		rcqs_abs(&abs_true_vec, &ctmp); // get_cqsvector_i(true_vec, i));
		if(rqs_cmp_ui(abs_true_vec.val, 0UL) != 0)
			rqs_div(abs_diff_vec.val, abs_diff_vec.val, abs_true_vec.val);
		
		if(rqs_cmp(max_relerr, abs_diff_vec.val) < 0)
			rqs_set(max_relerr, abs_diff_vec.val);
		if(rqs_cmp(min_relerr, abs_diff_vec.val) > 0)
			rqs_set(min_relerr, abs_diff_vec.val);
	}

	free_cqsvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_cqsmatrix(CQSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
    row_swap_qsmatrix(mat->re, row_index0, row_index1, col_start, col_end);
    row_swap_qsmatrix(mat->im, row_index0, row_index1, col_start, col_end);
}

// print cqsmatrix
void print_cqsmatrix(CQSMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->re->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->re->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_dd_write((vec->element + index * QSSIZE));
			//c_dd_write(GET_DDMATRIX_IJ(mat, row_index, col_index));
			rqs_out_str_base(stdout, 10, 64, get_cqsmatrix_ij_cqsfloat(mat, row_index, col_index).val_re);
            printf(" + ");
   			rqs_out_str_base(stdout, 10, 64, get_cqsmatrix_ij_cqsfloat(mat, row_index, col_index).val_im);
            printf(" * I\n");
		}
	}
}

// print cqsmatrix2
#if 0
void print_cqsmatrix2(CQSMatrix mat1, CQSMatrix mat2)
{
	long int row_index, col_index;
    long int row_dim, col_dim;

    row_dim = (mat1->re->row_dim <= mat2->re->row_dim) ? mat1->re->row_dim : mat2->re->row_dim;
    col_dim = (mat1->re->col_dim <= mat2->re->col_dim) ? mat1->re->col_dim : mat2->re->col_dim;

	for(row_index = 0; row_index < row_dim; row_index++)
	{
		for(col_index = 0; col_index < col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_dd_write((vec->element + index * QSSIZE));
			//c_dd_write(GET_DDMATRIX_IJ(mat, row_index, col_index));
			rqs_out_str_base(stdout, 10, 64, get_cqsmatrix_ij_cqsfloat(mat1, row_index, col_index).val_re);
            printf(", ");
			rqs_out_str_base(stdout, 10, 64, get_cqsmatrix_ij_cqsfloat(mat2, row_index, col_index).val_re);
            printf(" + ");
   			rqs_out_str_base(stdout, 10, 64, get_cqsmatrix_ij_cqsfloat(mat1, row_index, col_index).val_im);
            printf(", ");
			rqs_out_str_base(stdout, 10, 64, get_cqsmatrix_ij_cqsfloat(mat2, row_index, col_index).val_im);
            printf(" * I\n");
		}
	}
}
#endif // 0


#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus

#ifdef DEBUG
int main()
{
    qsfloat a, b, c
    cqsfloat ca, cb, cc;

    rqs_set_d(a->val, 3.0); rqs_sqrt(a->val);
    rqs_set_d(b->val, 5.0); rqs_sqrt(b->val);

    rqs_out_str_base(stdout, 10, 64, a);
    rqs_out_str_base(stdout, 10, 64, b);
    
}
#endif // DEBUG

