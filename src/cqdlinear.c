/********************************************************************************/
/* cqdlinear.c: Quadruple-double precision Complex Linear Computation Library      */
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
#include "cqdlinear.h"
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

// initialize CQDVector
CQDVector init_cqdvector(int dimension)
{
    CQDVector ret = NULL;

    ret = (CQDVector)malloc(sizeof(cqdvector));

    if(ret == NULL)
    {
        fprintf(stderr, "ERROR: init_cqdvector(%d)\n", dimension);
        return NULL;
    }

    ret->re = init_qdvector(dimension);
    if(ret->re == NULL)
    {
        fprintf(stderr, "ERROR: init_cqdvector(%d)\n", dimension);
        free(ret);
        return NULL;
    }

    ret->im = init_qdvector(dimension);
    if(ret->im == NULL)
    {
        fprintf(stderr, "ERROR: init_cqdvector(%d)\n", dimension);
        free(ret->re);
        free(ret);
        return NULL;
    }

    return ret;
}

// free CQDVector
void free_cqdvector(CQDVector vec)
{
    free_qdvector(vec->re);
    free_qdvector(vec->im);
    free(vec);
}

// CQDVector vec -> qdfloat array
void set_cqdfloat_cqdvec(cqdfloat ret[], int ret_dim, CQDVector vec)
{
    int i;
    cqdfloat ptr_val;

    for(i = 0; i < ret_dim; i++)
        subst_cqdvector_i(&ret[i], vec, i);
        //rcqd_set(&ret[i], get_cqdvector_i(vec, i));

/*    {
        ptr_val = get_cqdvector_i_cqdfloat(vec, i);
        rcqd_set(&ret[i], &ptr_val);
    }
*/

    return;
}

// cqdfloat array -> CQDVector ret
void set_cqdvector_cqdfloat(CQDVector ret, cqdfloat array[], int array_dim)
{
    int i;

    for(i = 0; i < ret->re->dim; i++)
        set_cqdvector_i(ret, i, &array[i]);
}

// qdvector -> cqdvector
void set_cqdvector_qdvec(CQDVector ret, QDVector re_vec, QDVector im_vec)
{
    subst_qdvector(ret->re, re_vec);
    subst_qdvector(ret->im, im_vec);
}

// print qdvector
void print_cqdvector(CQDVector vec)
{
    int i;

    for(i = 0; i < vec->re->dim; i++)
    {
        printf("%5d ", i);
        rqd_out_str(get_qdvector_i(vec->re, i));
        printf(" + ");
        rqd_out_str(get_qdvector_i(vec->im, i));
        printf("\n");
    }
}

// set a zero vector
void set0_cqdvector(CQDVector vec)
{
    set0_qdvector(vec->re);
    set0_qdvector(vec->im);
}

/*************************************************/
/* Vector Calculations for CQDVector               */
/*
void add_cqdvector(CQDVector c, CQDVector a, CQDVector b)
void add2_cqdvector(CQDVector c, CQDVector a)
void sub_cqdvector(CQDVector c, CQDVector a, CQDVector b)
void sub2_cqdvector(CQDVector c, DVector a)
void cmul_cqdvector(CQDVector c, double val[QDSIZE], CQDVector a)
void cmul2_cqdvector(CQDVector c, double val[QDSIZE])
void add_cmul_cqdvector(CQDVector c, CQDVector a, double val[QDSIZE], CQDVector b)
double ip_cqdvector(CQDVector a, CQDVector b)
double norm1_cqdvector(CQDVector a)
double norm2_cqdvector(CQDVector a)
double normi_cqdvector(CQDVector a)
void subst_cqdvector(CQDVector c, CQDVector a)
*/
/*************************************************/
/* c = a + b */
void add_cqdvector(CQDVector c, CQDVector a, CQDVector b)
{
    add_qdvector(c->re, a->re, b->re);
    add_qdvector(c->im, a->im, b->im);
}

/* c += a */
void add2_cqdvector(CQDVector c, CQDVector a)
{
    add2_qdvector(c->re, a->re);
    add2_qdvector(c->im, a->im);
}

/* c = a - b */
void sub_cqdvector(CQDVector c, CQDVector a, CQDVector b)
{
    sub_qdvector(c->re, a->re, b->re);
    sub_qdvector(c->im, a->im, b->im);
}

/* c -= a */
void sub2_cqdvector(CQDVector c, CQDVector a)
{
    sub2_qdvector(c->re, a->re);
    sub2_qdvector(c->im, a->im);
}

/* c = val * a */
void cmul_cqdvector_4m(CQDVector c, cqdfloat *val, CQDVector a)
{
    QDVector t1, t2, t3, t4;

    // 2024-11-28(Thu) Fixed! T.Kouya
    t1 = init_qdvector(c->re->dim);
    t2 = init_qdvector(c->re->dim);
    t3 = init_qdvector(c->re->dim);
    t4 = init_qdvector(c->re->dim);

    cmul_qdvector(t1, val->val_re, a->re);
    cmul_qdvector(t2, val->val_im, a->im);  
    cmul_qdvector(t3, val->val_im, a->re);
    cmul_qdvector(t4, val->val_re, a->im);

    sub_qdvector(c->re, t1, t2);
    add_qdvector(c->im, t3, t4);

    free_qdvector(t1);
    free_qdvector(t2);
    free_qdvector(t3);
    free_qdvector(t4);

}

/* c = val * a */
void cmul_cqdvector_3m(CQDVector c, cqdfloat *val, CQDVector a)
{
    QDVector t1, t2, t3;
    qdfloat tmp;

    t1 = init_qdvector(c->re->dim);
    t2 = init_qdvector(c->re->dim);

    cmul_qdvector(t1, val->val_re, a->re);
    cmul_qdvector(t2, val->val_im, a->im);
    sub_qdvector(c->re, t1, t2);

    //#ifdef USE_4M
        // 4M
    /*
        cmul_qdvector(t1, val->val_im, a->re);
        cmul_qdvector(t2, val->val_re, a->im);
        add_qdvector(c->im, t1, t2);
    */
    //#else // USE_4M
        // 3M
        rqd_add(tmp.val, val->val_re, val->val_im);
        t3 = init_qdvector(c->re->dim);
        add_qdvector(t3, a->re, a->im);
        cmul_qdvector(c->im, tmp.val, t3);
        sub_qdvector(c->im, c->im, t1);
        sub_qdvector(c->im, c->im, t2);
        free_qdvector(t3);
    //#endif // USE_4M

    free_qdvector(t1);
    free_qdvector(t2);
}


/* c *= val */
void cmul2_cqdvector(CQDVector c, cqdfloat *val)
{
    CQDVector in_a;

    in_a = init_cqdvector(c->re->dim);

    subst_cqdvector(in_a, c);
    cmul_cqdvector(c, val, in_a);

    free_cqdvector(in_a);
}

/* c = a + val * b */
void add_cmul_cqdvector(CQDVector c, CQDVector a, cqdfloat *val, CQDVector b)
{
    CQDVector in_vec;
    in_vec = init_cqdvector(b->re->dim);

    //cmul_cqdvector(c, val, b);
    cmul_cqdvector(in_vec, val, b);
    //add2_cqdvector(c, a);
    add_cqdvector(c, a, in_vec);

    free_cqdvector(in_vec);
}

/* c = a - val * b */
void sub_cmul_cqdvector(CQDVector c, CQDVector a, cqdfloat *val, CQDVector b)
{
    CQDVector in_vec;
    in_vec = init_cqdvector(b->re->dim);

    //cmul_cqdvector(c, val, b);
    cmul_cqdvector(in_vec, val, b);
    //add2_cqdvector(c, a);
    sub_cqdvector(c, a, in_vec);

    free_cqdvector(in_vec);
}

/* (a, b) = conj(a)^T * b */
void ip_cqdvector(cqdfloat *ret, CQDVector a, CQDVector b)
{
    int i;
    cqdfloat tmp, conj_a_i, b_i;

    rcqd_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cqdvector_i(&tmp, a, i);
        rcqd_conj(&conj_a_i, &tmp); //get_cqdvector_i(a, i));
        subst_cqdvector_i(&b_i, b, i);
        rcqd_mul(&tmp, &conj_a_i, &b_i); //get_cqdvector_i(b, i));
        rcqd_add(ret, ret, &tmp);
    }
}

/* a^T * b */
void dotp_cqdvector(cqdfloat *ret, CQDVector a, CQDVector b)
{
    int i;
    cqdfloat tmp, ai, bi;

    rcqd_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cqdvector_i(&ai, a, i);
        subst_cqdvector_i(&bi, b, i);
        rcqd_mul(&tmp, &ai, &bi); //get_cqdvector_i(a, i), get_cqdvector_i(b, i));
        rcqd_add(ret, ret, &tmp);
    }
}

/* c := a */
void subst_cqdvector(CQDVector c, CQDVector a)
{
    subst_qdvector(c->re, a->re);
    subst_qdvector(c->im, a->im);
}

/* c := conj(a) */
void conj_cqdvector(CQDVector c, CQDVector a)
{
    subst_qdvector(c->re, a->re);
    neg_qdvector(c->im, a->im);
}

/* c := -a */
void neg_cqdvector(CQDVector c, CQDVector a)
{
    neg_qdvector(c->re, a->re);
    neg_qdvector(c->im, a->im);
}

/* ||a||_1 */
void norm1_cqdvector(double ret[QDSIZE], CQDVector a)
{
    int i;
    qdfloat tmp;
    cqdfloat ctmp;

    rqd_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cqdvector_i(&ctmp, a, i);
        rcqd_abs(&tmp, &ctmp);
        rqd_add(ret, ret, tmp.val);
    }
}

/* ||a||_infty */
void normi_cqdvector(double ret[QDSIZE], CQDVector a)
{
    int i;
    qdfloat tmp;
    cqdfloat ctmp;

    rqd_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cqdvector_i(&ctmp, a, i);
        rcqd_abs(&tmp, &ctmp);
        if(rqd_cmp(ret, tmp.val) < 0)
            rqd_set(ret, tmp.val);
    }
}

// Euclid norm
void norm2_cqdvector(double ret[QDSIZE], CQDVector vec)
{
    int i;
    qdfloat tmp;
    cqdfloat ctmp;

    rqd_set0(ret);
    for(i = 0; i < vec->re->dim; i++)
    {
        subst_cqdvector_i(&ctmp, vec, i);
        rcqd_nrm2(&tmp, &ctmp);
        rqd_add(ret, ret, tmp.val);
    }
    #ifdef USE_GMP
    rqd_sqrt_mpfr(ret, ret);
    #else // USE_GMP
    rqd_sqrt(ret, ret);
    #endif // USE_GMP
}

// set a zero matrix
//void set0_cqdmatrix(CQDMatrix mat);
void set0_cqdmatrix(CQDMatrix mat)
{
    set0_qdmatrix(mat->re);
    set0_qdmatrix(mat->im);
}

// initialize qdvector
CQDMatrix init_cqdmatrix(long int row_dim, long int col_dim)
{
    CQDMatrix ret = NULL;

    ret = (CQDMatrix)malloc(sizeof(cqdmatrix));
    if(ret == NULL)
    {
        fprintf(stderr, "ERROR: init_cqdmatrix(%ld, %ld)\n", row_dim, col_dim);
        return NULL;
    }

    ret->re = init_qdmatrix(row_dim, col_dim);
    if(ret->re == NULL)
    {
        fprintf(stderr, "ERROR: init_cqdmatrix(%ld, %ld)\n", row_dim, col_dim);
        free(ret);
        return NULL;
    }

    ret->im = init_qdmatrix(row_dim, col_dim);
    if(ret->im == NULL)
    {
        fprintf(stderr, "ERROR: init_cqdmatrix(%ld, %ld)\n", row_dim, col_dim);
        free_qdmatrix(ret->re);
        free(ret);
        return NULL;
    }

    return ret;
}

// free csqdvector
void free_cqdmatrix(CQDMatrix mat)
{
    free_qdmatrix(mat->re);
    free_qdmatrix(mat->im);
    free(mat);
}

// print cqdvector
//void print_cqdmatrix(CQDMatrix mat);

// qdmatrix -> cqdmatrix
void set_cqdmatrix_qdmat(CQDMatrix ret, QDMatrix re_mat, QDMatrix im_mat)
{
    subst_qdmatrix(ret->re, re_mat);
    subst_qdmatrix(ret->im, im_mat);
}

// CQDMatrix mat -> cqdfloat array
void set_cqdfloat_cqdmat(cqdfloat ret[], int ret_dim, CQDMatrix mat)
{
    long int i, j, index;
    cqdfloat mat_ij;

    index = 0;
    for(i = 0; i < mat->re->row_dim; i++)
    {
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_cqdmatrix_ij(&mat_ij, mat, i, j); 
            rcqd_set(&ret[index++], &mat_ij); //get_cqdmatrix_ij(mat, i, j));
        }

    }

    return;
}

// qdfloat array -> DDmatrix ret
void set_cqdmatrix_cqdfloat(CQDMatrix ret, cqdfloat array[], int array_dim)
{
    long int i, j, index;

    index = 0;
    for(i = 0; i < ret->re->row_dim; i++)
    {
        for(j = 0; j < ret->re->col_dim; j++)
            set_cqdmatrix_ij(ret, i, j, &array[index++]);
    }

    return;
}

// matrix multiplication
// ret := A * B
void mul_cqdmatrix_4m(CQDMatrix ret, CQDMatrix a, CQDMatrix b)
{
    QDMatrix t1, t2, t3, t4;

    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_qdmatrix(t1, a->re, b->re);
    mul_qdmatrix(t2, a->im, b->im);
    sub_qdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_qdmatrix(t3, a->im, b->re);
        mul_qdmatrix(t4, a->re, b->im);
        add_qdmatrix(ret->im, t3, t4);
    //#else // USE_4M 
        // 3M
    /*
        add_qdmatrix(t3, a->re, a->im);
        add_qdmatrix(t4, b->re, b->im);
        mul_qdmatrix(ret->im, t3, t4);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    */
    //#endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// matrix multiplication
// ret := A * B
void mul_cqdmatrix_3m(CQDMatrix ret, CQDMatrix a, CQDMatrix b)
{
    QDMatrix t1, t2, t3, t4;

    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_qdmatrix(b->re->row_dim, ret->re->col_dim);

    mul_qdmatrix(t1, a->re, b->re);
    mul_qdmatrix(t2, a->im, b->im);
    sub_qdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
    /*
        mul_qdmatrix(t3, a->im, b->re);
        mul_qdmatrix(t4, a->re, b->im);
        add_qdmatrix(ret->im, t1, t2);
    */
    //#else // USE_4M 
        // 3M
        add_qdmatrix(t3, a->re, a->im);
        add_qdmatrix(t4, b->re, b->im);
        mul_qdmatrix(ret->im, t3, t4);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// Frobenius norm
void normf_cqdmatrix(double ret[QDSIZE], CQDMatrix mat)
{
    int i, j;
    qdfloat tmp;
    cqdfloat ctmp;

    rqd_set0(ret);
    for(i = 0; i < mat->re->row_dim; i++)
    {
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_cqdmatrix_ij(&ctmp, mat, i, j);
            rcqd_nrm2(&tmp, &ctmp);
            rqd_add(ret, ret, tmp.val);
        }
    }
    rqd_sqrt(ret, ret);
}

// print normf
void print_normf_cqdmatrix(const char *str, CQDMatrix mat)
{
    qdfloat tmp;

    normf_cqdmatrix(tmp.val, mat);

    rqd_out_str(tmp.val);
}

/*************************************************/
/* Matrix Caluculations for CQDMatrix            */
/*
void normf_cqdmatrix(double ret[QDSIZE], CQDMatrix mat)
void norm1_cqdmatrix(double ret[QDSIZE], CQDMatrix mat)
void normi_cqdmatrix(double ret[QDSIZE], CQDMatrix mat)
void add_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b);
void sub_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b);
void mul_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b);
void mul_cqdmatrix_cqdvec(CQDVector v, CQDMatrix a, CQDVector vb)
void mul_cqdmatrixt_cqdvec(CQDVector v, CQDMatrix a, CQDVector vb)
void transpose_cqdmatrix(CQDMatrix c, CQDMatrix a);
void inv_cqdmatrix(CQDMatrix a);
void subst_mpfmatrux(CQDMatrix c, CQDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_cqdmatrix(double ret[QDSIZE], CQDMatrix mat)
{
    int i, j;
    qdfloat tmp, sum;
    cqdfloat ctmp;

    rqd_set0(ret);
    for(j = 0; j < mat->re->col_dim; j++)
    {
        rqd_set0(sum.val);
        for(i = 0; i < mat->re->row_dim; i++)
        {
            subst_cqdmatrix_ij(&ctmp, mat, i, j);
            rcqd_abs(&tmp, &ctmp);
            rqd_add(sum.val, sum.val, tmp.val);
        }
        if(rqd_cmp(ret, sum.val) < 0)
            rqd_set(ret, sum.val);
    }
}

/* 1 Norm of Matrix */
void norm1_cqdmatrix(double ret[QDSIZE], CQDMatrix mat)
{
    int i, j;
    qdfloat tmp, sum;
    cqdfloat ctmp;

    rqd_set0(ret);
    for(i = 0; i < mat->re->row_dim; i++)
    {
        rqd_set0(sum.val);
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_cqdmatrix_ij(&ctmp, mat, i, j);
            rcqd_abs(&tmp, &ctmp);
            rqd_add(sum.val, sum.val, tmp.val);
        }
        if(rqd_cmp(ret, sum.val) < 0)
            rqd_set(ret, sum.val);
    }
}

/* c := a + b */
void add_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b)
{
    add_qdmatrix(c->re, a->re, b->re);
    add_qdmatrix(c->im, a->im, b->im);
}

/* c := a - b */
void sub_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b)
{
    sub_qdmatrix(c->re, a->re, b->re);
    sub_qdmatrix(c->im, a->im, b->im);
}

/* c := sc * a */
void cmul_cqdmatrix(CQDMatrix c, cqdfloat *sc, CQDMatrix a)
{
    QDMatrix t1, t2, t3;
    qdfloat tmp;

    t1 = init_qdmatrix(c->re->row_dim, c->re->col_dim);
    t2 = init_qdmatrix(c->re->row_dim, c->re->col_dim);

    cmul_qdmatrix(t1, sc->val_re, a->re);
    cmul_qdmatrix(t2, sc->val_im, a->im);
    sub_qdmatrix(c->re, t1, t2);

    #ifdef USE_4M
        // 4M
        cmul_qdmatrix(t1, sc->val_im, a->re);
        cmul_qdmatrix(t2, sc->val_re, a->im);
        add_qdmatrix(c->im, t1, t2);
    #else // USE_4M
        // 3M
        rqd_add(tmp.val, sc->val_re, sc->val_im);
        t3 = init_qdmatrix(c->re->row_dim, c->im->col_dim);
        add_qdmatrix(t3, a->re, a->im);
        cmul_qdmatrix(c->im, tmp.val, t3);
        sub_qdmatrix(c->im, c->im, t1);
        sub_qdmatrix(c->im, c->im, t2);
        free_qdmatrix(t3);
    #endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
}

/* c = a^T */
void transpose_cqdmatrix(CQDMatrix c, CQDMatrix a)
{
    transpose_qdmatrix(c->re, a->re);
    transpose_qdmatrix(c->im, a->im);
}

/* c := conj(a)^T */
void star_cqdmatrix(CQDMatrix c, CQDMatrix a)
{
    // c_re := -a_im
    neg_qdmatrix(c->re, a->im);
    // c_im := -a_im^T
    transpose_qdmatrix(c->im, c->re);
    // c_re := a_re^T
    transpose_qdmatrix(c->re, a->re);
}

/* c := conj(a) */
void conj_cqdmatrix(CQDMatrix c, CQDMatrix a)
{
    subst_qdmatrix(c->re, a->re);
    neg_qdmatrix(c->im, a->im);
}

/* c := -a */
void neg_cqdmatrix(CQDMatrix c, CQDMatrix a)
{
    neg_qdmatrix(c->re, a->re);
    neg_qdmatrix(c->im, a->im);
}

/* c := a */
void subst_cqdmatrix(CQDMatrix c, CQDMatrix a)
{
    subst_qdmatrix(c->re, a->re);
    subst_qdmatrix(c->im, a->im);
}

/* c := I */
void setI_cqdmatrix(CQDMatrix c)
{
    setI_qdmatrix(c->re);
    set0_qdmatrix(c->im);
}

/* v := a * vb */
void mul_cqdmatrix_cqdvec_4m(CQDVector v, CQDMatrix a, CQDVector vb)
{
    QDVector t1, t2, t3, t4;
    QDMatrix tmp_mat;

    t1 = init_qdvector(v->re->dim);
    t2 = init_qdvector(v->re->dim);
    t3 = init_qdvector(v->re->dim);
    t4 = init_qdvector(v->re->dim);

    mul_qdmatrix_qdvec(t1, a->re, vb->re);
    mul_qdmatrix_qdvec(t2, a->im, vb->im);
    sub_qdvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        mul_qdmatrix_qdvec(t3, a->im, vb->re);
        mul_qdmatrix_qdvec(t4, a->re, vb->im);
        add_qdvector(v->im, t3, t4);
    //#else // USE_4M
    /*
        // 3M
        tmp_mat = init_qdmatrix(a->re->row_dim, a->re->col_dim);
        add_qdmatrix(tmp_mat, a->re, a->im);
        add_qdvector(t3, vb->re, vb->im);
        mul_qdmatrix_qdvec(t4, tmp_mat, t3);
        sub_qdvector(v->im, t4, t1);
        sub_qdvector(v->im, v->im, t2);
        free_qdmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_qdvector(t1);
    free_qdvector(t2);
    free_qdvector(t3);
    free_qdvector(t4);
}

/* v := a * vb */
void mul_cqdmatrix_cqdvec_3m(CQDVector v, CQDMatrix a, CQDVector vb)
{
    QDVector t1, t2, t3, t4;
    QDMatrix tmp_mat;

    t1 = init_qdvector(v->re->dim);
    t2 = init_qdvector(v->re->dim);
    t3 = init_qdvector(v->re->dim);
    t4 = init_qdvector(v->re->dim);

    mul_qdmatrix_qdvec(t1, a->re, vb->re);
    mul_qdmatrix_qdvec(t2, a->im, vb->im);
    sub_qdvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
    /*
        mul_qdmatrix_qdvec(t3, a->im, vb->re);
        mul_qdmatrix_qdvec(t4, a->re, vb->im);
        add_qdvector(v->im, t3, t4);
    */
    //#else // USE_4M
        // 3M
        tmp_mat = init_qdmatrix(a->re->row_dim, a->re->col_dim);
        add_qdmatrix(tmp_mat, a->re, a->im);
        add_qdvector(t3, vb->re, vb->im);
        mul_qdmatrix_qdvec(t4, tmp_mat, t3);
        sub_qdvector(v->im, t4, t1);
        sub_qdvector(v->im, v->im, t2);
        free_qdmatrix(tmp_mat);
    //#endif // USE_4M

    free_qdvector(t1);
    free_qdvector(t2);
    free_qdvector(t3);
    free_qdvector(t4);
}

/* v := a^T * vb */
void mul_cqdmatrixt_cqdvec(CQDVector v, CQDMatrix a, CQDVector vb)
{
    QDVector t1, t2, t3, t4;
    QDMatrix tmp_mat;

    t1 = init_qdvector(v->re->dim);
    t2 = init_qdvector(v->re->dim);
    t3 = init_qdvector(v->re->dim);
    t4 = init_qdvector(v->re->dim);

    mul_qdmatrixt_qdvec(t1, a->re, vb->re);
    mul_qdmatrixt_qdvec(t2, a->im, vb->im);
    sub_qdvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        mul_qdmatrixt_qdvec(t3, a->im, vb->re);
        mul_qdmatrixt_qdvec(t4, a->re, vb->im);
        add_qdvector(v->im, t3, t4);
    //#else // USE_4M
        // 3M
    /*
        tmp_mat = init_qdmatrix(a->re->row_dim, a->re->col_dim);
        add_qdmatrix(tmp_mat, a->re, a->im);
        add_qdvector(t3, vb->re, vb->im);
        mul_qdmatrixt_qdvec(t4, tmp_mat, t3);
        sub_qdvector(v->im, t4, t1);
        sub_qdvector(v->im, v->im, t2);
        free_qdmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_qdvector(t1);
    free_qdvector(t2);
    free_qdvector(t3);
    free_qdvector(t4);
}

/* v := conj(a)^T * vb */
void mul_cqdmatrixs_cqdvec(CQDVector v, CQDMatrix a, CQDVector vb)
{
    QDVector t1, t2, t3, t4;
    QDMatrix tmp_mat;

    t1 = init_qdvector(v->re->dim);
    t2 = init_qdvector(v->re->dim);
    t3 = init_qdvector(v->re->dim);
    t4 = init_qdvector(v->re->dim);

    mul_qdmatrixt_qdvec(t1, a->re, vb->re);
    mul_qdmatrixt_qdvec(t2, a->im, vb->im);
    add_qdvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        mul_qdmatrixt_qdvec(t3, a->im, vb->re);
        mul_qdmatrixt_qdvec(t4, a->re, vb->im);
        sub_qdvector(v->im, t3, t4);
    //#else // USE_4M
        // 3M
    /*
        tmp_mat = init_qdmatrix(a->re->row_dim, a->re->col_dim);
        add_qdmatrix(tmp_mat, a->re, a->im);
        add_qdvector(t3, vb->re, vb->im);
        mul_qdmatrixt_qdvec(t4, tmp_mat, t3);
        sub_qdvector(v->im, t4, t1);
        sub_qdvector(v->im, v->im, t2);
        free_qdmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_qdvector(t1);
    free_qdvector(t2);
    free_qdvector(t3);
    free_qdvector(t4);
}

/* a = a^(-1) */
/* square matrix only */
void inv_cqdmatrix(CQDMatrix a)
{
	long int i, j, k, dim;
	cqdfloat ctmp, aii, aij, aik, ajk, aji;
	qdfloat tmp;

	/* Check Dimensions */
	if(a->re->row_dim != a->re->col_dim)
	{
		fprintf(stderr, "ERROR: inv_cqdmatrix\n");
		return;
	}

	dim = a->re->row_dim;

	for(i = 0; i < dim; i++)
	{
        subst_cqdmatrix_ij(&aii, a, i, i);
		rcqd_abs(&tmp, &aii); //get_cqdmatrix_ij(a, i, i));
		if(rqd_cmp_ui(tmp.val, 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_cqdmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		rcqd_inv(&aii, &aii); //get_cqdmatrix_ij(a, i, i));
		//inv_mpc_t(aii, get_cmpfmatrix_ij(a, i, i));
		set_cqdmatrix_ij(a, i, i, &aii);

		for(j = 0; j < i; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
            subst_cqdmatrix_ij(&aij, a, i, j);
			rcqd_mul(&ctmp, &aij, &aii); //get_cqdmatrix_ij(a, i, j), &aii);
			set_cqdmatrix_ij(a, i, j, &ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
            subst_cqdmatrix_ij(&aij, a, i, j);
			rcqd_mul(&ctmp, &aij, &aii); //get_cqdmatrix_ij(a, i, j), &aii);
			set_cqdmatrix_ij(a, i, j, &ctmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cqdmatrix_ij(&aji, a, j, i);
                subst_cqdmatrix_ij(&aik, a, i, k);
				subst_cqdmatrix_ij(&ajk, a, j, k);
				rcqd_mul(&ctmp, &aji, &aik); //get_cqdmatrix_ij(a, i, k));
				rcqd_sub(&ctmp, &ajk, &ctmp);
				set_cqdmatrix_ij(a, j, k, &ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cqdmatrix_ij(&aji, a, j, i);
                subst_cqdmatrix_ij(&aik, a, i, k);
				subst_cqdmatrix_ij(&ajk, a, j, k);
				rcqd_mul(&ctmp, &aji, &aik); //get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, k));
				rcqd_sub(&ctmp, &ajk, &ctmp); //get_cqdmatrix_ij(a, j, k), &ctmp);
				set_cqdmatrix_ij(a, j, k, &ctmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cqdmatrix_ij(&aji, a, j, i);
                subst_cqdmatrix_ij(&aik, a, i, k);
				subst_cqdmatrix_ij(&ajk, a, j, k);
				rcqd_mul(&ctmp, &aji, &aik); //get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, k));
				rcqd_sub(&ctmp, &ajk, &ctmp); //get_cqdmatrix_ij(a, j, k), &ctmp);
				set_cqdmatrix_ij(a, j, k, &ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cqdmatrix_ij(&aji, a, j, i);
                subst_cqdmatrix_ij(&aik, a, i, k);
				subst_cqdmatrix_ij(&ajk, a, j, k);
				rcqd_mul(&ctmp, &aji, &aik); //get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, k));
				rcqd_sub(&ctmp, &ajk, &ctmp); //get_cqdmatrix_ij(a, j, k), &ctmp);
				set_cqdmatrix_ij(a, j, k, &ctmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			rcqd_neg(&ctmp, &aii);
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
            subst_cqdmatrix_ij(&aji, a, j, i);
			rcqd_mul(&ctmp, &ctmp, &aji); //get_cqdmatrix_ij(a, j, i));
			set_cqdmatrix_ij(a, j, i, &ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			rcqd_neg(&ctmp, &aii);
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
            subst_cqdmatrix_ij(&aji, a, j, i);
			rcqd_mul(&ctmp, &ctmp, &aji); //get_cqdmatrix_ij(a, j, i));
			set_cqdmatrix_ij(a, j, i, &ctmp);
		}
	}
}

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpc)a */
void subst_cmpfvector_cqdvec(CMPFVector c, CQDVector a)
{
    MPFVector re, im;

    re = init2_mpfvector(c->dim, c->prec);
    im = init2_mpfvector(c->dim, c->prec);

    subst_mpfvector_qdvec(re, a->re);
    subst_mpfvector_qdvec(im, a->im);

    merge_cmpfvector(c, re, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (dd)a */
void subst_cqdvector_cmpfvec(CQDVector c, CMPFVector a)
{
    MPFVector re, im;

    re = init2_mpfvector(a->dim, a->prec);
    im = init2_mpfvector(a->dim, a->prec);

    separate_cmpfvector(re, im, a);
    subst_qdvector_mpfvec(c->re, re);
    subst_qdvector_mpfvec(c->im, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (mpf)a */
void subst_cmpfmatrix_cqdmat(CMPFMatrix c, CQDMatrix a)
{
    MPFMatrix re, im;

    re = init2_mpfmatrix(c->row_dim, c->col_dim, c->prec);
    im = init2_mpfmatrix(c->row_dim, c->col_dim, c->prec);

    subst_mpfmatrix_qdmat(re, a->re);
    subst_mpfmatrix_qdmat(im, a->im);

    merge_cmpfmatrix(c, re, im);

    free_mpfmatrix(re);
    free_mpfmatrix(im);
}


/* c := (dd)a */
void subst_cqdmatrix_cmpfmat(CQDMatrix c, CMPFMatrix a)
{
    MPFMatrix re, im;

    re = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    im = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);

    separate_cmpfmatrix(re, im, a);
    subst_qdmatrix_mpfmat(c->re, re);
    subst_qdmatrix_mpfmat(c->im, im);

    free_mpfmatrix(re);
    free_mpfmatrix(im);
}

/* Normwise relative error of vector */
void relerr_cqdvector_cmpfvec(double relerr[QDSIZE], CQDVector approx_vec, CMPFVector true_vec, int norm_type)
{}

/* Elementwise relative errors of vector */
void relerr_element_cqdvector_mpf(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], CQDVector approx_vec, MPFVector true_vec, int norm_type)
{}
#endif // USE_GMP

/* c := (dd)a */
void subst_cqdvector_cdvec(CQDVector c, CDVector a)
{
    DVector re, im;

    re = init_dvector(a->dim);
    im = init_dvector(a->dim);

    separate_cdvector(re, im, a);
    subst_qdvector_dvec(c->re, re);
    subst_qdvector_dvec(c->im, im);

    free_dvector(re);
    free_dvector(im);
}

/* c := (d)a */
void subst_cdvector_cqdvec(CDVector c, CQDVector a)
{
    DVector re, im;

    re = init_dvector(a->re->dim);
    im = init_dvector(a->im->dim);

    subst_dvector_qdvec(re, a->re);
    subst_dvector_qdvec(im, a->im);

    merge_cdvector(c, re, im);

    free_dvector(re);
    free_dvector(im);
}

/* c := (dd)a */
void subst_cqdmatrix_cdmat(CQDMatrix c, CDMatrix a)
{
    DMatrix re, im;

    re = init_dmatrix(a->row_dim, a->col_dim);
    im = init_dmatrix(a->row_dim, a->col_dim);

    separate_cdmatrix(re, im, a);
    subst_qdmatrix_dmat(c->re, re);
    subst_qdmatrix_dmat(c->im, im);

    free_dmatrix(re);
    free_dmatrix(im);
}

/* c := (d)a */
void subst_cdmatrix_cqdmat(CDMatrix c, CQDMatrix a)
{
    DMatrix re, im;

    re = init_dmatrix(c->row_dim, c->col_dim);
    im = init_dmatrix(c->row_dim, c->col_dim);

    subst_dmatrix_qdmat(re, a->re);
    subst_dmatrix_qdmat(im, a->im);
    merge_cdmatrix(c, re, im);

    free_dmatrix(re);
    free_dmatrix(im);
}


/* c := (qd)a */
void subst_cqdvector_cddvec(CQDVector c, CDDVector a)
{
    subst_qdvector_ddvec(c->re, a->re);
    subst_qdvector_ddvec(c->im, a->im);
}

/* c := (dd)a */
void subst_cddvector_cqdvec(CDDVector c, CQDVector a)
{
    subst_ddvector_qdvec(c->re, a->re);
    subst_ddvector_qdvec(c->im, a->im);
}

/* c := (qd)a */
void subst_cqdmatrix_cddmat(CQDMatrix c, CDDMatrix a)
{
    subst_qdmatrix_ddmat(c->re, a->re);
    subst_qdmatrix_ddmat(c->im, a->im);
}

/* c := (dd)a */
void subst_cddmatrix_cqdmat(CDDMatrix c, CQDMatrix a)
{
    subst_ddmatrix_qdmat(c->re, a->re);
    subst_ddmatrix_qdmat(c->im, a->im);
}



/* c := (qd)a */
void subst_cqdvector_ctdvec(CQDVector c, CTDVector a)
{
    subst_qdvector_tdvec(c->re, a->re);
    subst_qdvector_tdvec(c->im, a->im);
}

/* c := (td)a */
void subst_ctdvector_cqdvec(CTDVector c, CQDVector a)
{
    subst_tdvector_qdvec(c->re, a->re);
    subst_tdvector_qdvec(c->im, a->im);
}

/* c := (qd)a */
void subst_cqdmatrix_ctdmat(CQDMatrix c, CTDMatrix a)
{
    subst_qdmatrix_tdmat(c->re, a->re);
    subst_qdmatrix_tdmat(c->im, a->im);
}

/* c := (td)a */
void subst_ctdmatrix_cqdmat(CTDMatrix c, CQDMatrix a)
{
    subst_tdmatrix_qdmat(c->re, a->re);
    subst_tdmatrix_qdmat(c->im, a->im);
}



/* Normwise relative error of vector */
void relerr_cqdvector(double relerr[QDSIZE], CQDVector approx_vec, CQDVector true_vec, int norm_type)
{
    double norm_true_vec[QDSIZE], norm_diff_vec[QDSIZE];
	CQDVector diff_vec;

	diff_vec = init_cqdvector(approx_vec->re->dim);

	// diff_vec := approx_vec - true_vec
	sub_cqdvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_cqdvector(norm_diff_vec, diff_vec);
			normi_cqdvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_cqdvector(norm_diff_vec, diff_vec);
			norm1_cqdvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_cqdvector(norm_diff_vec, diff_vec);
			norm2_cqdvector(norm_true_vec, true_vec);
			break;
	}

	if(rqd_cmp_ui(norm_true_vec, 0UL) != 0)
		rqd_div(relerr, norm_diff_vec, norm_true_vec);

	free_cqdvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_cqdvector(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], CQDVector approx_vec, CQDVector true_vec, int norm_type)
{
    qdfloat abs_true_vec, abs_diff_vec;
    double norm_diff_vec[QDSIZE], norm_true_vec[QDSIZE];
	long int i;
	CQDVector diff_vec;
    cqdfloat ctmp;

	diff_vec = init_cqdvector(approx_vec->re->dim);

	// diff_vec := approx_vec - true_vec
	sub_cqdvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_cqdvector(norm_diff_vec, diff_vec);
			normi_cqdvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_cqdvector(norm_diff_vec, diff_vec);
			norm1_cqdvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_cqdvector(norm_diff_vec, diff_vec);
			norm2_cqdvector(norm_true_vec, true_vec);
			break;
	}

	rqd_set(norm_relerr, norm_diff_vec);
	if(rqd_cmp_ui(norm_true_vec, 0UL) != 0)
		rqd_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rqd_set_ui(max_relerr, 0UL);
	normi_cqdvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->re->dim; i++)
	{
        subst_cqdvector_i(&ctmp, diff_vec, i);
		rcqd_abs(&abs_diff_vec, &ctmp); // get_cqdvector_i(diff_vec, i));
        subst_cqdvector_i(&ctmp, true_vec, i);
		rcqd_abs(&abs_true_vec, &ctmp); // get_cqdvector_i(true_vec, i));
		if(rqd_cmp_ui(abs_true_vec.val, 0UL) != 0)
			rqd_div(abs_diff_vec.val, abs_diff_vec.val, abs_true_vec.val);
		
		if(rqd_cmp(max_relerr, abs_diff_vec.val) < 0)
			rqd_set(max_relerr, abs_diff_vec.val);
		if(rqd_cmp(min_relerr, abs_diff_vec.val) > 0)
			rqd_set(min_relerr, abs_diff_vec.val);
	}

	free_cqdvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_cqdmatrix(CQDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
    row_swap_qdmatrix(mat->re, row_index0, row_index1, col_start, col_end);
    row_swap_qdmatrix(mat->im, row_index0, row_index1, col_start, col_end);
}

// print cqdmatrix
void print_cqdmatrix(CQDMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->re->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->re->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_dd_write((vec->element + index * QDSIZE));
			//c_dd_write(GET_DDMATRIX_IJ(mat, row_index, col_index));
			rqd_out_str(get_cqdmatrix_ij_cqdfloat(mat, row_index, col_index).val_re);
            printf(" + ");
   			rqd_out_str(get_cqdmatrix_ij_cqdfloat(mat, row_index, col_index).val_im);
            printf(" * I\n");
		}
	}
}

// print cqdmatrix2
#if 0
void print_cqdmatrix2(CQDMatrix mat1, CQDMatrix mat2)
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
	//		c_dd_write((vec->element + index * QDSIZE));
			//c_dd_write(GET_DDMATRIX_IJ(mat, row_index, col_index));
			rqd_out_str(get_cqdmatrix_ij_cqdfloat(mat1, row_index, col_index).val_re);
            printf(", ");
			rqd_out_str(get_cqdmatrix_ij_cqdfloat(mat2, row_index, col_index).val_re);
            printf(" + ");
   			rqd_out_str(get_cqdmatrix_ij_cqdfloat(mat1, row_index, col_index).val_im);
            printf(", ");
			rqd_out_str(get_cqdmatrix_ij_cqdfloat(mat2, row_index, col_index).val_im);
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
    qdfloat a, b, c
    cqdfloat ca, cb, cc;

    rqd_set_d(a->val, 3.0); rqd_sqrt(a->val);
    rqd_set_d(b->val, 5.0); rqd_sqrt(b->val);

    rqd_out_str(a);
    rqd_out_str(b);
    
}
#endif // DEBUG

