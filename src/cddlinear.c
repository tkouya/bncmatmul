/********************************************************************************/
/* cddlinear.c: Double-double precision Complex Linear Computation Library      */
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
#include "cddlinear.h"
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

// initialize CDDVector
CDDVector init_cddvector(int dimension)
{
    CDDVector ret = NULL;

    ret = (CDDVector)malloc(sizeof(cddvector));

    if(ret == NULL)
    {
        fprintf(stderr, "ERROR: init_cddvector(%d)\n", dimension);
        return NULL;
    }

    ret->re = init_ddvector(dimension);
    if(ret->re == NULL)
    {
        fprintf(stderr, "ERROR: init_cddvector(%d)\n", dimension);
        free(ret);
        return NULL;
    }

    ret->im = init_ddvector(dimension);
    if(ret->im == NULL)
    {
        fprintf(stderr, "ERROR: init_cddvector(%d)\n", dimension);
        free(ret->re);
        free(ret);
        return NULL;
    }

    return ret;
}

// free CDDVector
void free_cddvector(CDDVector vec)
{
    free_ddvector(vec->re);
    free_ddvector(vec->im);
    free(vec);
}

// CDDVector vec -> cddfloat array
void set_cddfloat_cddvec(cddfloat ret[], int ret_dim, CDDVector vec)
{
    int i;
    cddfloat ptr_val;

    for(i = 0; i < ret_dim; i++)
    {
        subst_cddvector_i(&ptr_val, vec, i);
        rcdd_set(&ret[i], &ptr_val); // get_cddvector_i(vec, i));
    }


/*    {
        ptr_val = get_cddvector_i_cddfloat(vec, i);
        rcdd_set(&ret[i], &ptr_val);
    }
*/

    return;
}

// ddvector -> cddvector
void set_cddvector_ddvec(CDDVector ret, DDVector re_vec, DDVector im_vec)
{
    subst_ddvector(ret->re, re_vec);
    subst_ddvector(ret->im, im_vec);
}

// cddfloat array -> CDDVector ret
void set_cddvector_cddfloat(CDDVector ret, cddfloat array[], int array_dim)
{
    int i;

    for(i = 0; i < ret->re->dim; i++)
        set_cddvector_i(ret, i, &array[i]);
}

// print ddvector
void print_cddvector(CDDVector vec)
{
    long int i;
    cddfloat ret;

    for(i = 0; i < vec->re->dim; i++)
    {
        printf("%5ld ", i);
        // fixed!: 2024-09-23 T.Kouya
/*        rdd_out_str(get_ddvector_i(vec->re, i));
        printf(" + ");
        rdd_out_str(get_ddvector_i(vec->im, i));
        printf(" * I\n");
*/
        //ret = get_cddvector_i_cddfloat(vec, i);
        subst_cddvector_i(&ret, vec, i);
        //rdd_out_str(get_cddvector_i_cddfloat(vec, i).val_re);
        rdd_out_str(ret.val_re);
        printf(" + ");
   		//rdd_out_str(get_cddvector_i_cddfloat(vec, i).val_im);
        rdd_out_str(ret.val_im);
        printf(" * I\n");

    }
}

// set a zero vector
void set0_cddvector(CDDVector vec)
{
    set0_ddvector(vec->re);
    set0_ddvector(vec->im);
}

/*************************************************/
/* Vector Calculations for CDDVector               */
/*
void add_cddvector(CDDVector c, CDDVector a, CDDVector b)
void add2_cddvector(CDDVector c, CDDVector a)
void sub_cddvector(CDDVector c, CDDVector a, CDDVector b)
void sub2_cddvector(CDDVector c, DVector a)
void cmul_cddvector(CDDVector c, double val[DDSIZE], CDDVector a)
void cmul2_cddvector(CDDVector c, double val[DDSIZE])
void add_cmul_cddvector(CDDVector c, CDDVector a, double val[DDSIZE], CDDVector b)
double ip_cddvector(CDDVector a, CDDVector b)
double norm1_cddvector(CDDVector a)
double norm2_cddvector(CDDVector a)
double normi_cddvector(CDDVector a)
void subst_cddvector(CDDVector c, CDDVector a)
*/
/*************************************************/
/* c = a + b */
void add_cddvector(CDDVector c, CDDVector a, CDDVector b)
{
    add_ddvector(c->re, a->re, b->re);
    add_ddvector(c->im, a->im, b->im);
}

/* c += a */
void add2_cddvector(CDDVector c, CDDVector a)
{
    add2_ddvector(c->re, a->re);
    add2_ddvector(c->im, a->im);
}

/* c = a - b */
void sub_cddvector(CDDVector c, CDDVector a, CDDVector b)
{
    sub_ddvector(c->re, a->re, b->re);
    sub_ddvector(c->im, a->im, b->im);
}

/* c -= a */
void sub2_cddvector(CDDVector c, CDDVector a)
{
    sub2_ddvector(c->re, a->re);
    sub2_ddvector(c->im, a->im);
}

/* c = val * a */
void cmul_cddvector_4m(CDDVector c, cddfloat *val, CDDVector a)
{
    DDVector t1, t2, t3, t4;
    ddfloat tmp;

    // 2024-11-28(Thu) Fixed! T.Kouya
    t1 = init_ddvector(c->re->dim);
    t2 = init_ddvector(c->re->dim);
    t3 = init_ddvector(c->re->dim);
    t4 = init_ddvector(c->re->dim);

    cmul_ddvector(t1, val->val_re, a->re);
    cmul_ddvector(t2, val->val_im, a->im);
    cmul_ddvector(t3, val->val_im, a->re);
    cmul_ddvector(t4, val->val_re, a->im);

    sub_ddvector(c->re, t1, t2);
    add_ddvector(c->im, t3, t4);

    free_ddvector(t1);
    free_ddvector(t2);
    free_ddvector(t3);
    free_ddvector(t4);
}

/* c = val * a */
void cmul_cddvector_3m(CDDVector c, cddfloat *val, CDDVector a)
{
    DDVector t1, t2, t3;
    ddfloat tmp;

    t1 = init_ddvector(c->re->dim);
    t2 = init_ddvector(c->re->dim);

    cmul_ddvector(t1, val->val_re, a->re);
    cmul_ddvector(t2, val->val_im, a->im);
    sub_ddvector(c->re, t1, t2);

    //#ifdef USE_4M
        // 4M
    /*
        cmul_ddvector(t1, val->val_im, a->re);
        cmul_ddvector(t2, val->val_re, a->im);
        add_ddvector(c->im, t1, t2);
    */
    //#else // USE_4M
        // 3M
        rdd_add(tmp.val, val->val_re, val->val_im);
        t3 = init_ddvector(c->re->dim);
        add_ddvector(t3, a->re, a->im);
        cmul_ddvector(c->im, tmp.val, t3);
        sub_ddvector(c->im, c->im, t1);
        sub_ddvector(c->im, c->im, t2);
        free_ddvector(t3);
    //#endif // USE_4M

    free_ddvector(t1);
    free_ddvector(t2);
}


/* c *= val */
void cmul2_cddvector(CDDVector c, cddfloat *val)
{
    CDDVector in_a;

    in_a = init_cddvector(c->re->dim);

    subst_cddvector(in_a, c);
    cmul_cddvector(c, val, in_a);

    free_cddvector(in_a);
}

/* c = a + val * b */
void add_cmul_cddvector(CDDVector c, CDDVector a, cddfloat *val, CDDVector b)
{
    CDDVector in_vec;
    in_vec = init_cddvector(b->re->dim);

    //cmul_cddvector(c, val, b);
    cmul_cddvector(in_vec, val, b);
    //add2_cddvector(c, a);
    add_cddvector(c, a, in_vec);

    free_cddvector(in_vec);
}

/* c = a - val * b */
void sub_cmul_cddvector(CDDVector c, CDDVector a, cddfloat *val, CDDVector b)
{
    CDDVector in_vec;
    in_vec = init_cddvector(b->re->dim);

    //cmul_cddvector(c, val, b);
    cmul_cddvector(in_vec, val, b);
    //sub2_cddvector(c, a);
    sub_cddvector(c, a, in_vec);

    free_cddvector(in_vec);
}

/* (a, b) = conj(a)^T * b */
void ip_cddvector(cddfloat *ret, CDDVector a, CDDVector b)
{
    int i;
    cddfloat tmp, conj_a_i, ai, bi;

    rcdd_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cddvector_i(&ai, a, i);
        rcdd_conj(&conj_a_i, &ai); // get_cddvector_i(a, i));
        subst_cddvector_i(&bi, b, i);
        rcdd_mul(&tmp, &conj_a_i, &bi); // get_cddvector_i(b, i));
        rcdd_add(ret, ret, &tmp);
    }
}

/* a^T * b */
void dotp_cddvector(cddfloat *ret, CDDVector a, CDDVector b)
{
    int i;
    cddfloat tmp, ai, bi;

    rcdd_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cddvector_i(&ai, a, i);
        subst_cddvector_i(&bi, b, i);
        rcdd_mul(&tmp, &ai, &bi); // get_cddvector_i(a, i), get_cddvector_i(b, i));
        rcdd_add(ret, ret, &tmp);
    }
}

/* c := a */
void subst_cddvector(CDDVector c, CDDVector a)
{
    subst_ddvector(c->re, a->re);
    subst_ddvector(c->im, a->im);
}

/* c := conj(a) */
void conj_cddvector(CDDVector c, CDDVector a)
{
    subst_ddvector(c->re, a->re);
    neg_ddvector(c->im, a->im);
}

/* c := -a */
void neg_cddvector(CDDVector c, CDDVector a)
{
    neg_ddvector(c->re, a->re);
    neg_ddvector(c->im, a->im);
}

/* ||a||_1 */
void norm1_cddvector(double ret[DDSIZE], CDDVector a)
{
    int i;
    ddfloat tmp;
    cddfloat ai;

    rdd_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cddvector_i(&ai, a, i);
        rcdd_abs(&tmp, &ai); // get_cddvector_i(a, i));
        rdd_add(ret, ret, tmp.val);
    }
}

/* ||a||_infty */
void normi_cddvector(double ret[DDSIZE], CDDVector a)
{
    int i;
    ddfloat tmp;
    cddfloat ai;

    rdd_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cddvector_i(&ai, a, i);
        rcdd_abs(&tmp, &ai); // get_cddvector_i(a, i));
        if(rdd_cmp(ret, tmp.val) < 0)
            rdd_set(ret, tmp.val);
    }
}

// Euclid norm
void norm2_cddvector(double ret[DDSIZE], CDDVector vec)
{
    int i;
    ddfloat tmp;
    cddfloat vec_i;

    rdd_set0(ret);
    for(i = 0; i < vec->re->dim; i++)
    {
        subst_cddvector_i(&vec_i, vec, i);
        rcdd_nrm2(&tmp, &vec_i); // get_cddvector_i(vec, i));
        rdd_add(ret, ret, tmp.val);
    }
    rdd_sqrt(ret, ret);
}

// set a zero matrix
//void set0_cddmatrix(CDDMatrix mat);
void set0_cddmatrix(CDDMatrix mat)
{
    set0_ddmatrix(mat->re);
    set0_ddmatrix(mat->im);
}

// initialize ddvector
CDDMatrix init_cddmatrix(long int row_dim, long int col_dim)
{
    CDDMatrix ret = NULL;

    ret = (CDDMatrix)malloc(sizeof(cddmatrix));
    if(ret == NULL)
    {
        fprintf(stderr, "ERROR: init_cddmatrix(%ld, %ld)\n", row_dim, col_dim);
        return NULL;
    }

    ret->re = init_ddmatrix(row_dim, col_dim);
    if(ret->re == NULL)
    {
        fprintf(stderr, "ERROR: init_cddmatrix(%ld, %ld)\n", row_dim, col_dim);
        free(ret);
        return NULL;
    }

    ret->im = init_ddmatrix(row_dim, col_dim);
    if(ret->im == NULL)
    {
        fprintf(stderr, "ERROR: init_cddmatrix(%ld, %ld)\n", row_dim, col_dim);
        free_ddmatrix(ret->re);
        free(ret);
        return NULL;
    }

    return ret;
}

// free csddvector
void free_cddmatrix(CDDMatrix mat)
{
    free_ddmatrix(mat->re);
    free_ddmatrix(mat->im);
    free(mat);
}

// print cddvector
//void print_cddmatrix(CDDMatrix mat);

// CDDMatrix mat -> cddfloat array
void set_cddfloat_cddmat(cddfloat ret[], int ret_dim, CDDMatrix mat)
{
    long int i, j, index;
    cddfloat mat_ij;

    index = 0;
    for(i = 0; i < mat->re->row_dim; i++)
    {
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_cddmatrix_ij(&mat_ij, mat, i, j);
            rcdd_set(&ret[index++], &mat_ij); // get_cddmatrix_ij(mat, i, j));
        }
    }

    return;
}

// ddmatrix -> cddmatrix
void set_cddmatrix_ddmat(CDDMatrix ret, DDMatrix re_mat, DDMatrix im_mat)
{
    subst_ddmatrix(ret->re, re_mat);
    subst_ddmatrix(ret->im, im_mat);
}

// cddfloat array -> CDDmatrix ret
void set_cddmatrix_cddfloat(CDDMatrix ret, cddfloat array[], int array_dim)
{
    long int i, j, index;

    index = 0;
    for(i = 0; i < ret->re->row_dim; i++)
    {
        for(j = 0; j < ret->re->col_dim; j++)
            set_cddmatrix_ij(ret, i, j, &array[index++]);
    }

    return;
}

// matrix multiplication
// ret := A * B
void mul_cddmatrix_4m(CDDMatrix ret, CDDMatrix a, CDDMatrix b)
{
    DDMatrix t1, t2, t3, t4;

    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_ddmatrix(t1, a->re, b->re);
    mul_ddmatrix(t2, a->im, b->im);
    sub_ddmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_ddmatrix(t3, a->im, b->re);
        mul_ddmatrix(t4, a->re, b->im);
        add_ddmatrix(ret->im, t3, t4);
    //#else // USE_4M 
        // 3M
    /*
        add_ddmatrix(t3, a->re, a->im);
        add_ddmatrix(t4, b->re, b->im);
        mul_ddmatrix(ret->im, t3, t4);
        sub_ddmatrix(ret->im, ret->im, t1);
        sub_ddmatrix(ret->im, ret->im, t2);
    */
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// matrix multiplication
// ret := A * B
void mul_cddmatrix_3m(CDDMatrix ret, CDDMatrix a, CDDMatrix b)
{
    DDMatrix t1, t2, t3, t4;

    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_ddmatrix(b->re->row_dim, ret->re->col_dim);

    mul_ddmatrix(t1, a->re, b->re);
    mul_ddmatrix(t2, a->im, b->im);
    sub_ddmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
    /*
        mul_ddmatrix(t3, a->im, b->re);
        mul_ddmatrix(t4, a->re, b->im);
        add_ddmatrix(ret->im, t1, t2);
    */
    //#else // USE_4M 
        // 3M
        add_ddmatrix(t3, a->re, a->im);
        add_ddmatrix(t4, b->re, b->im);
        mul_ddmatrix(ret->im, t3, t4);
        sub_ddmatrix(ret->im, ret->im, t1);
        sub_ddmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}


// Frobenius norm
void normf_cddmatrix(double ret[DDSIZE], CDDMatrix mat)
{
    int i, j;
    ddfloat tmp;
    cddfloat mat_ij;

    rdd_set0(ret);
    for(i = 0; i < mat->re->row_dim; i++)
    {
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_cddmatrix_ij(&mat_ij, mat, i, j);
            rcdd_nrm2(&tmp, &mat_ij); // get_cddmatrix_ij(mat, i, j));
            rdd_add(ret, ret, tmp.val);
        }
    }
    rdd_sqrt(ret, ret);
}

// print normf
void print_normf_cddmatrix(const char *str, CDDMatrix mat)
{
    ddfloat tmp;

    normf_cddmatrix(tmp.val, mat);

    rdd_out_str(tmp.val);
}

/*************************************************/
/* Matrix Caluculations for CDDMatrix            */
/*
void normf_cddmatrix(double ret[DDSIZE], CDDMatrix mat)
void norm1_cddmatrix(double ret[DDSIZE], CDDMatrix mat)
void normi_cddmatrix(double ret[DDSIZE], CDDMatrix mat)
void add_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b);
void sub_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b);
void mul_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b);
void mul_cddmatrix_ddvec(CDDVector v, CDDMatrix a, CDDVector vb)
void mul_cddmatrixt_ddvec(CDDVector v, CDDMatrix a, CDDVector vb)
void transpose_cddmatrix(CDDMatrix c, CDDMatrix a);
void inv_cddmatrix(CDDMatrix a);
void subst_mpfmatrux(CDDMatrix c, CDDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_cddmatrix(double ret[DDSIZE], CDDMatrix mat)
{
    int i, j;
    ddfloat tmp, sum;
    cddfloat mat_ij;

    rdd_set0(ret);
    for(j = 0; j < mat->re->col_dim; j++)
    {
        rdd_set0(sum.val);
        for(i = 0; i < mat->re->row_dim; i++)
        {
            subst_cddmatrix_ij(&mat_ij, mat, i, j);
            rcdd_abs(&tmp, &mat_ij); // get_cddmatrix_ij(mat, i, j));
            rdd_add(sum.val, sum.val, tmp.val);
        }
        if(rdd_cmp(ret, sum.val) < 0)
            rdd_set(ret, sum.val);
    }
}

/* 1 Norm of Matrix */
void norm1_cddmatrix(double ret[DDSIZE], CDDMatrix mat)
{
    int i, j;
    ddfloat tmp, sum;
    cddfloat mat_ij;

    rdd_set0(ret);
    for(i = 0; i < mat->re->row_dim; i++)
    {
        rdd_set0(sum.val);
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_cddmatrix_ij(&mat_ij, mat, i, j);
            rcdd_abs(&tmp, &mat_ij); // get_cddmatrix_ij(mat, i, j));
            rdd_add(sum.val, sum.val, tmp.val);
        }
        if(rdd_cmp(ret, sum.val) < 0)
            rdd_set(ret, sum.val);
    }
}

/* c := a + b */
void add_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b)
{
    add_ddmatrix(c->re, a->re, b->re);
    add_ddmatrix(c->im, a->im, b->im);
}

/* c := a - b */
void sub_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b)
{
    sub_ddmatrix(c->re, a->re, b->re);
    sub_ddmatrix(c->im, a->im, b->im);
}

/* c := sc * a */
void cmul_cddmatrix(CDDMatrix c, cddfloat *sc, CDDMatrix a)
{
    DDMatrix t1, t2, t3;
    ddfloat tmp;

    t1 = init_ddmatrix(c->re->row_dim, c->re->col_dim);
    t2 = init_ddmatrix(c->re->row_dim, c->re->col_dim);

    cmul_ddmatrix(t1, sc->val_re, a->re);
    cmul_ddmatrix(t2, sc->val_im, a->im);
    sub_ddmatrix(c->re, t1, t2);

    #ifdef USE_4M
        // 4M
        cmul_ddmatrix(t1, sc->val_im, a->re);
        cmul_ddmatrix(t2, sc->val_re, a->im);
        add_ddmatrix(c->im, t1, t2);
    #else // USE_4M
        // 3M
        rdd_add(tmp.val, sc->val_re, sc->val_im);
        t3 = init_ddmatrix(c->re->row_dim, c->im->col_dim);
        add_ddmatrix(t3, a->re, a->im);
        cmul_ddmatrix(c->im, tmp.val, t3);
        sub_ddmatrix(c->im, c->im, t1);
        sub_ddmatrix(c->im, c->im, t2);
        free_ddmatrix(t3);
    #endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
}


/* c = a^T */
void transpose_cddmatrix(CDDMatrix c, CDDMatrix a)
{
    transpose_ddmatrix(c->re, a->re);
    transpose_ddmatrix(c->im, a->im);
}

/* c := conj(a)^T */
void star_cddmatrix(CDDMatrix c, CDDMatrix a)
{
    // c_re := -a_im
    neg_ddmatrix(c->re, a->im);
    // c_im := -a_im^T
    transpose_ddmatrix(c->im, c->re);
    // c_re := a_re^T
    transpose_ddmatrix(c->re, a->re);
}

/* c := a */
void subst_cddmatrix(CDDMatrix c, CDDMatrix a)
{
    subst_ddmatrix(c->re, a->re);
    subst_ddmatrix(c->im, a->im);
}

/* c := conj(a) */
void conj_cddmatrix(CDDMatrix c, CDDMatrix a)
{
    subst_ddmatrix(c->re, a->re);
    neg_ddmatrix(c->im, a->im);
}

/* c := -a */
void neg_cddmatrix(CDDMatrix c, CDDMatrix a)
{
    neg_ddmatrix(c->re, a->re);
    neg_ddmatrix(c->im, a->im);
}

/* c := I */
void setI_cddmatrix(CDDMatrix c)
{
    setI_ddmatrix(c->re);
    set0_ddmatrix(c->im);
}

/* v := a * vb */
void mul_cddmatrix_cddvec_4m(CDDVector v, CDDMatrix a, CDDVector vb)
{
    DDVector t1, t2, t3, t4;
    DDMatrix tmp_mat;

    t1 = init_ddvector(v->re->dim);
    t2 = init_ddvector(v->re->dim);
    t3 = init_ddvector(v->re->dim);
    t4 = init_ddvector(v->re->dim);

    mul_ddmatrix_ddvec(t1, a->re, vb->re);
    mul_ddmatrix_ddvec(t2, a->im, vb->im);
    sub_ddvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        mul_ddmatrix_ddvec(t3, a->im, vb->re);
        mul_ddmatrix_ddvec(t4, a->re, vb->im);
        add_ddvector(v->im, t3, t4);
    //#else // USE_4M
    /*
        // 3M
        tmp_mat = init_ddmatrix(a->re->row_dim, a->re->col_dim);
        add_ddmatrix(tmp_mat, a->re, a->im);
        add_ddvector(t3, vb->re, vb->im);
        mul_ddmatrix_ddvec(t4, tmp_mat, t3);
        sub_ddvector(v->im, t4, t1);
        sub_ddvector(v->im, v->im, t2);
        free_ddmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_ddvector(t1);
    free_ddvector(t2);
    free_ddvector(t3);
    free_ddvector(t4);
}

/* v := a * vb */
void mul_cddmatrix_cddvec_3m(CDDVector v, CDDMatrix a, CDDVector vb)
{
    DDVector t1, t2, t3, t4;
    DDMatrix tmp_mat;

    t1 = init_ddvector(v->re->dim);
    t2 = init_ddvector(v->re->dim);
    t3 = init_ddvector(v->re->dim);
    t4 = init_ddvector(v->re->dim);

    mul_ddmatrix_ddvec(t1, a->re, vb->re);
    mul_ddmatrix_ddvec(t2, a->im, vb->im);
    sub_ddvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
    /*
        mul_ddmatrix_ddvec(t3, a->im, vb->re);
        mul_ddmatrix_ddvec(t4, a->re, vb->im);
        add_ddvector(v->im, t3, t4);
    */
    //#else // USE_4M
        // 3M
        tmp_mat = init_ddmatrix(a->re->row_dim, a->re->col_dim);
        add_ddmatrix(tmp_mat, a->re, a->im);
        add_ddvector(t3, vb->re, vb->im);
        mul_ddmatrix_ddvec(t4, tmp_mat, t3);
        sub_ddvector(v->im, t4, t1);
        sub_ddvector(v->im, v->im, t2);
        free_ddmatrix(tmp_mat);
    //#endif // USE_4M

    free_ddvector(t1);
    free_ddvector(t2);
    free_ddvector(t3);
    free_ddvector(t4);
}


/* v := a^T * vb */
void mul_cddmatrixt_cddvec(CDDVector v, CDDMatrix a, CDDVector vb)
{
    DDVector t1, t2, t3, t4;
    DDMatrix tmp_mat;

    t1 = init_ddvector(v->re->dim);
    t2 = init_ddvector(v->re->dim);
    t3 = init_ddvector(v->re->dim);
    t4 = init_ddvector(v->re->dim);

    mul_ddmatrixt_ddvec(t1, a->re, vb->re);
    mul_ddmatrixt_ddvec(t2, a->im, vb->im);
    sub_ddvector(v->re, t1, t2);

   // #ifdef USE_4M
        // 4M
        mul_ddmatrixt_ddvec(t3, a->im, vb->re);
        mul_ddmatrixt_ddvec(t4, a->re, vb->im);
        add_ddvector(v->im, t3, t4);
    //#else // USE_4M
    /*
        // 3M
        tmp_mat = init_ddmatrix(a->re->row_dim, a->re->col_dim);
        add_ddmatrix(tmp_mat, a->re, a->im);
        add_ddvector(t3, vb->re, vb->im);
        mul_ddmatrixt_ddvec(t4, tmp_mat, t3);
        sub_ddvector(v->im, t4, t1);
        sub_ddvector(v->im, v->im, t2);
        free_ddmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_ddvector(t1);
    free_ddvector(t2);
    free_ddvector(t3);
    free_ddvector(t4);
}

/* v := conj(a)^T * vb */
void mul_cddmatrixs_cddvec(CDDVector v, CDDMatrix a, CDDVector vb)
{
    DDVector t1, t2, t3, t4;
    DDMatrix tmp_mat;

    t1 = init_ddvector(v->re->dim);
    t2 = init_ddvector(v->re->dim);
    t3 = init_ddvector(v->re->dim);
    t4 = init_ddvector(v->re->dim);

    mul_ddmatrixt_ddvec(t1, a->re, vb->re);
    mul_ddmatrixt_ddvec(t2, a->im, vb->im);
    add_ddvector(v->re, t1, t2);

   // #ifdef USE_4M
        // 4M
        mul_ddmatrixt_ddvec(t3, a->im, vb->re);
        mul_ddmatrixt_ddvec(t4, a->re, vb->im);
        sub_ddvector(v->im, t3, t4);
    //#else // USE_4M
    /*
        // 3M
        tmp_mat = init_ddmatrix(a->re->row_dim, a->re->col_dim);
        add_ddmatrix(tmp_mat, a->re, a->im);
        add_ddvector(t3, vb->re, vb->im);
        mul_ddmatrixt_ddvec(t4, tmp_mat, t3);
        sub_ddvector(v->im, t4, t1);
        sub_ddvector(v->im, v->im, t2);
        free_ddmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_ddvector(t1);
    free_ddvector(t2);
    free_ddvector(t3);
    free_ddvector(t4);
}


/* a = a^(-1) */
/* square matrix only */
void inv_cddmatrix(CDDMatrix a)
{
	long int i, j, k, dim;
	cddfloat ctmp, aii, aij, aik, ajk, aji;
	ddfloat tmp;

	/* Check Dimensions */
	if(a->re->row_dim != a->re->col_dim)
	{
		fprintf(stderr, "ERROR: inv_cddmatrix\n");
		return;
	}

	dim = a->re->row_dim;

	for(i = 0; i < dim; i++)
	{
		rcdd_abs(&tmp, get_cddmatrix_ij(a, i, i));
		if(rdd_cmp_ui(tmp.val, 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_cddmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

        subst_cddmatrix_ij(&ctmp, a, i, i);
		rcdd_inv(&aii, &ctmp); // get_cddmatrix_ij(a, i, i));
		//inv_mpc_t(aii, get_cmpfmatrix_ij(a, i, i));
		set_cddmatrix_ij(a, i, i, &aii);

		for(j = 0; j < i; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
            subst_cddmatrix_ij(&aij, a, i, j);
			rcdd_mul(&ctmp, &aij, &aii); // get_cddmatrix_ij(a, i, j), &aii);
			set_cddmatrix_ij(a, i, j, &ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
            subst_cddmatrix_ij(&aij, a, i, j);
			rcdd_mul(&ctmp, &aij, &aii); // get_cddmatrix_ij(a, i, j), &aii);
			set_cddmatrix_ij(a, i, j, &ctmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cddmatrix_ij(&aji, a, j, i);
                subst_cddmatrix_ij(&aik, a, i, k);
                subst_cddmatrix_ij(&ajk, a, j, k);
				rcdd_mul(&ctmp, &aji, &aik); // get_cddmatrix_ij(a, j, i), get_cddmatrix_ij(a, i, k));
				rcdd_sub(&ctmp, &ajk, &ctmp); // get_cddmatrix_ij(a, j, k), &ctmp);
				set_cddmatrix_ij(a, j, k, &ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cddmatrix_ij(&aji, a, j, i);
                subst_cddmatrix_ij(&aik, a, i, k);
                subst_cddmatrix_ij(&ajk, a, j, k);
				rcdd_mul(&ctmp, &aji, &aik); // get_cddmatrix_ij(a, j, i), get_cddmatrix_ij(a, i, k));
				rcdd_sub(&ctmp, &ajk, &ctmp); // get_cddmatrix_ij(a, j, k), &ctmp);
				set_cddmatrix_ij(a, j, k, &ctmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cddmatrix_ij(&aji, a, j, i);
                subst_cddmatrix_ij(&aik, a, i, k);
                subst_cddmatrix_ij(&ajk, a, j, k);
				rcdd_mul(&ctmp, &aji, &aik); // get_cddmatrix_ij(a, j, i), get_cddmatrix_ij(a, i, k));
				rcdd_sub(&ctmp, &ajk, &ctmp); // get_cddmatrix_ij(a, j, k), &ctmp);
				set_cddmatrix_ij(a, j, k, &ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cddmatrix_ij(&aji, a, j, i);
                subst_cddmatrix_ij(&aik, a, i, k);
                subst_cddmatrix_ij(&ajk, a, j, k);
				rcdd_mul(&ctmp, &aji, &aik); // get_cddmatrix_ij(a, j, i), get_cddmatrix_ij(a, i, k));
				rcdd_sub(&ctmp, &ajk, &ctmp); // get_cddmatrix_ij(a, j, k), &ctmp);
				set_cddmatrix_ij(a, j, k, &ctmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			rcdd_neg(&ctmp, &aii);
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
            subst_cddmatrix_ij(&aji, a, j, i);
			rcdd_mul(&ctmp, &ctmp, &aji); // get_cddmatrix_ij(a, j, i));
			set_cddmatrix_ij(a, j, i, &ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			rcdd_neg(&ctmp, &aii);
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
            subst_cddmatrix_ij(&aji, a, j, i);
			rcdd_mul(&ctmp, &ctmp, &aji); // get_cddmatrix_ij(a, j, i));
			set_cddmatrix_ij(a, j, i, &ctmp);
		}
	}
}

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpc)a */
void subst_cmpfvector_cddvec(CMPFVector c, CDDVector a)
{
    MPFVector re, im;

    re = init2_mpfvector(c->dim, c->prec);
    im = init2_mpfvector(c->dim, c->prec);

    subst_mpfvector_ddvec(re, a->re);
    subst_mpfvector_ddvec(im, a->im);

    merge_cmpfvector(c, re, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (dd)a */
void subst_cddvector_cmpfvec(CDDVector c, CMPFVector a)
{
    MPFVector re, im;

    re = init2_mpfvector(a->dim, a->prec);
    im = init2_mpfvector(a->dim, a->prec);

    separate_cmpfvector(re, im, a);
    subst_ddvector_mpfvec(c->re, re);
    subst_ddvector_mpfvec(c->im, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (dd)a / max|a_i| */
void subst_coef_cddvector_cmpfvec(CDDVector c, CMPFVector a)
{
    MPFVector re, im;
    mpc_t absmax_a;

    re = init2_mpfvector(a->dim, a->prec);
    im = init2_mpfvector(a->dim, a->prec);

    separate_cmpfvector(re, im, a);
    subst_ddvector_mpfvec(c->re, re);
    subst_ddvector_mpfvec(c->im, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (mpf)a */
void subst_cmpfmatrix_cddmat(CMPFMatrix c, CDDMatrix a)
{
    MPFMatrix re, im;

    re = init2_mpfmatrix(c->row_dim, c->col_dim, c->prec);
    im = init2_mpfmatrix(c->row_dim, c->col_dim, c->prec);

    subst_mpfmatrix_ddmat(re, a->re);
    subst_mpfmatrix_ddmat(im, a->im);

    merge_cmpfmatrix(c, re, im);

    free_mpfmatrix(re);
    free_mpfmatrix(im);
}


/* c := (dd)a */
void subst_cddmatrix_cmpfmat(CDDMatrix c, CMPFMatrix a)
{
    MPFMatrix re, im;

    re = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    im = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);

    separate_cmpfmatrix(re, im, a);
    subst_ddmatrix_mpfmat(c->re, re);
    subst_ddmatrix_mpfmat(c->im, im);

    free_mpfmatrix(re);
    free_mpfmatrix(im);
}

/* Normwise relative error of vector */
void relerr_cddvector_cmpfvec(double relerr[DDSIZE], CDDVector approx_vec, CMPFVector true_vec, int norm_type)
{}

/* Elementwise relative errors of vector */
void relerr_element_cddvector_mpf(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], CDDVector approx_vec, MPFVector true_vec, int norm_type)
{}
#endif // USE_GMP

/* c := (dd)a */
void subst_cddvector_cdvec(CDDVector c, CDVector a)
{
    DVector re, im;

    re = init_dvector(a->dim);
    im = init_dvector(a->dim);

    separate_cdvector(re, im, a);
    subst_ddvector_dvec(c->re, re);
    subst_ddvector_dvec(c->im, im);

    free_dvector(re);
    free_dvector(im);
}

/* c := (d)a */
void subst_cdvector_cddvec(CDVector c, CDDVector a)
{
    DVector re, im;

    re = init_dvector(a->re->dim);
    im = init_dvector(a->im->dim);

    subst_dvector_ddvec(re, a->re);
    subst_dvector_ddvec(im, a->im);

    merge_cdvector(c, re, im);

    free_dvector(re);
    free_dvector(im);
}

/* c := (dd)a */
void subst_cddmatrix_cdmat(CDDMatrix c, CDMatrix a)
{
    DMatrix re, im;

    re = init_dmatrix(a->row_dim, a->col_dim);
    im = init_dmatrix(a->row_dim, a->col_dim);

    separate_cdmatrix(re, im, a);
    subst_ddmatrix_dmat(c->re, re);
    subst_ddmatrix_dmat(c->im, im);

    free_dmatrix(re);
    free_dmatrix(im);
}

/* c := (d)a */
void subst_cdmatrix_cddmat(CDMatrix c, CDDMatrix a)
{
    DMatrix re, im;

    re = init_dmatrix(c->row_dim, c->col_dim);
    im = init_dmatrix(c->row_dim, c->col_dim);

    subst_dmatrix_ddmat(re, a->re);
    subst_dmatrix_ddmat(im, a->im);
    merge_cdmatrix(c, re, im);

    free_dmatrix(re);
    free_dmatrix(im);
}

/* Normwise relative error of vector */
void relerr_cddvector(double relerr[DDSIZE], CDDVector approx_vec, CDDVector true_vec, int norm_type)
{
    double norm_true_vec[DDSIZE], norm_diff_vec[DDSIZE];
	CDDVector diff_vec;

	diff_vec = init_cddvector(approx_vec->re->dim);

	// diff_vec := approx_vec - true_vec
	sub_cddvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_cddvector(norm_diff_vec, diff_vec);
			normi_cddvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_cddvector(norm_diff_vec, diff_vec);
			norm1_cddvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_cddvector(norm_diff_vec, diff_vec);
			norm2_cddvector(norm_true_vec, true_vec);
			break;
	}

	if(rdd_cmp_ui(norm_true_vec, 0UL) != 0)
		rdd_div(relerr, norm_diff_vec, norm_true_vec);

	free_cddvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_cddvector(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], CDDVector approx_vec, CDDVector true_vec, int norm_type)
{
    ddfloat abs_true_vec, abs_diff_vec;
    double norm_diff_vec[DDSIZE], norm_true_vec[DDSIZE];
	long int i;
	CDDVector diff_vec;
    cddfloat ctmp;

	diff_vec = init_cddvector(approx_vec->re->dim);

	// diff_vec := approx_vec - true_vec
	sub_cddvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_cddvector(norm_diff_vec, diff_vec);
			normi_cddvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_cddvector(norm_diff_vec, diff_vec);
			norm1_cddvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_cddvector(norm_diff_vec, diff_vec);
			norm2_cddvector(norm_true_vec, true_vec);
			break;
	}

	rdd_set(norm_relerr, norm_diff_vec);
	if(rdd_cmp_ui(norm_true_vec, 0UL) != 0)
		rdd_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rdd_set_ui(max_relerr, 0UL);
	normi_cddvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->re->dim; i++)
	{
        subst_cddvector_i(&ctmp, diff_vec, i);
		rcdd_abs(&abs_diff_vec, &ctmp); // get_cddvector_i(diff_vec, i));
        subst_cddvector_i(&ctmp, true_vec, i);
		rcdd_abs(&abs_true_vec, &ctmp); // get_cddvector_i(true_vec, i));
		if(rdd_cmp_ui(abs_true_vec.val, 0UL) != 0)
			rdd_div(abs_diff_vec.val, abs_diff_vec.val, abs_true_vec.val);
		
		if(rdd_cmp(max_relerr, abs_diff_vec.val) < 0)
			rdd_set(max_relerr, abs_diff_vec.val);
		if(rdd_cmp(min_relerr, abs_diff_vec.val) > 0)
			rdd_set(min_relerr, abs_diff_vec.val);
	}

	free_cddvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_cddmatrix(CDDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
    row_swap_ddmatrix(mat->re, row_index0, row_index1, col_start, col_end);
    row_swap_ddmatrix(mat->im, row_index0, row_index1, col_start, col_end);
}

// print cddmatrix
void print_cddmatrix(CDDMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->re->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->re->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_dd_write((vec->element + index * DDSIZE));
			//c_dd_write(GET_DDMATRIX_IJ(mat, row_index, col_index));
			rdd_out_str(get_cddmatrix_ij_cddfloat(mat, row_index, col_index).val_re);
            printf(" + ");
   			rdd_out_str(get_cddmatrix_ij_cddfloat(mat, row_index, col_index).val_im);
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
    ddfloat a, b, c
    cddfloat ca, cb, cc;

    rdd_set_d(a->val, 3.0); rdd_sqrt(a->val);
    rdd_set_d(b->val, 5.0); rdd_sqrt(b->val);

    rdd_out_str(a);
    rdd_out_str(b);
    
}
#endif // DEBUG

