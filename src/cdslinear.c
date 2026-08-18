/********************************************************************************/
/* cddlinear.c: Double-float precision Complex Linear Computation Library      */
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
#include "cdslinear.h"
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

// file-local: real DS matrix negation (no neg_dsmatrix in single-precision lib)
static void _cds_neg_dsmatrix(DSMatrix c, DSMatrix a)
{
    float neg_one[DSSIZE];
    int i;
    neg_one[0] = -1.0f; for(i = 1; i < DSSIZE; i++) neg_one[i] = 0.0f;
    cmul_dsmatrix(c, neg_one, a);
}

// initialize CDSVector
CDSVector init_cdsvector(int dimension)
{
    CDSVector ret = NULL;

    ret = (CDSVector)malloc(sizeof(cdsvector));

    if(ret == NULL)
    {
        fprintf(stderr, "ERROR: init_cdsvector(%d)\n", dimension);
        return NULL;
    }

    ret->re = init_dsvector(dimension);
    if(ret->re == NULL)
    {
        fprintf(stderr, "ERROR: init_cdsvector(%d)\n", dimension);
        free(ret);
        return NULL;
    }

    ret->im = init_dsvector(dimension);
    if(ret->im == NULL)
    {
        fprintf(stderr, "ERROR: init_cdsvector(%d)\n", dimension);
        free(ret->re);
        free(ret);
        return NULL;
    }

    return ret;
}

// free CDSVector
void free_cdsvector(CDSVector vec)
{
    free_dsvector(vec->re);
    free_dsvector(vec->im);
    free(vec);
}

// CDSVector vec -> cdsfloat array
void set_cdsfloat_cdsvec(cdsfloat ret[], int ret_dim, CDSVector vec)
{
    int i;
    cdsfloat ptr_val;

    for(i = 0; i < ret_dim; i++)
    {
        subst_cdsvector_i(&ptr_val, vec, i);
        rcds_set(&ret[i], &ptr_val); // get_cdsvector_i(vec, i));
    }


/*    {
        ptr_val = get_cdsvector_i_cdsfloat(vec, i);
        rcds_set(&ret[i], &ptr_val);
    }
*/

    return;
}

// dsvector -> cdsvector
void set_cdsvector_dsvec(CDSVector ret, DSVector re_vec, DSVector im_vec)
{
    subst_dsvector(ret->re, re_vec);
    subst_dsvector(ret->im, im_vec);
}

// cdsfloat array -> CDSVector ret
void set_cdsvector_cdsfloat(CDSVector ret, cdsfloat array[], int array_dim)
{
    int i;

    for(i = 0; i < ret->re->dim; i++)
        set_cdsvector_i(ret, i, &array[i]);
}

// print dsvector
void print_cdsvector(CDSVector vec)
{
    long int i;
    cdsfloat ret;

    for(i = 0; i < vec->re->dim; i++)
    {
        printf("%5ld ", i);
        // fixed!: 2024-09-23 T.Kouya
/*        rds_out_str_base(stdout, 10, 33, get_dsvector_i(vec->re, i));
        printf(" + ");
        rds_out_str_base(stdout, 10, 33, get_dsvector_i(vec->im, i));
        printf(" * I\n");
*/
        //ret = get_cdsvector_i_cdsfloat(vec, i);
        subst_cdsvector_i(&ret, vec, i);
        //rds_out_str_base(stdout, 10, 33, get_cdsvector_i_cdsfloat(vec, i).val_re);
        rds_out_str_base(stdout, 10, 33, ret.val_re);
        printf(" + ");
   		//rds_out_str_base(stdout, 10, 33, get_cdsvector_i_cdsfloat(vec, i).val_im);
        rds_out_str_base(stdout, 10, 33, ret.val_im);
        printf(" * I\n");

    }
}

// set a zero vector
void set0_cdsvector(CDSVector vec)
{
    set0_dsvector(vec->re);
    set0_dsvector(vec->im);
}

/*************************************************/
/* Vector Calculations for CDSVector               */
/*
void add_cdsvector(CDSVector c, CDSVector a, CDSVector b)
void add2_cdsvector(CDSVector c, CDSVector a)
void sub_cdsvector(CDSVector c, CDSVector a, CDSVector b)
void sub2_cdsvector(CDSVector c, DVector a)
void cmul_cdsvector(CDSVector c, float val[DSSIZE], CDSVector a)
void cmul2_cdsvector(CDSVector c, float val[DSSIZE])
void add_cmul_cdsvector(CDSVector c, CDSVector a, float val[DSSIZE], CDSVector b)
float ip_cdsvector(CDSVector a, CDSVector b)
float norm1_cdsvector(CDSVector a)
float norm2_cdsvector(CDSVector a)
float normi_cdsvector(CDSVector a)
void subst_cdsvector(CDSVector c, CDSVector a)
*/
/*************************************************/
/* c = a + b */
void add_cdsvector(CDSVector c, CDSVector a, CDSVector b)
{
    add_dsvector(c->re, a->re, b->re);
    add_dsvector(c->im, a->im, b->im);
}

/* c += a */
void add2_cdsvector(CDSVector c, CDSVector a)
{
    add2_dsvector(c->re, a->re);
    add2_dsvector(c->im, a->im);
}

/* c = a - b */
void sub_cdsvector(CDSVector c, CDSVector a, CDSVector b)
{
    sub_dsvector(c->re, a->re, b->re);
    sub_dsvector(c->im, a->im, b->im);
}

/* c -= a */
void sub2_cdsvector(CDSVector c, CDSVector a)
{
    sub2_dsvector(c->re, a->re);
    sub2_dsvector(c->im, a->im);
}

/* c = val * a */
void cmul_cdsvector_4m(CDSVector c, cdsfloat *val, CDSVector a)
{
    DSVector t1, t2, t3, t4;
    dsfloat tmp;

    // 2024-11-28(Thu) Fixed! T.Kouya
    t1 = init_dsvector(c->re->dim);
    t2 = init_dsvector(c->re->dim);
    t3 = init_dsvector(c->re->dim);
    t4 = init_dsvector(c->re->dim);

    cmul_dsvector(t1, val->val_re, a->re);
    cmul_dsvector(t2, val->val_im, a->im);
    cmul_dsvector(t3, val->val_im, a->re);
    cmul_dsvector(t4, val->val_re, a->im);

    sub_dsvector(c->re, t1, t2);
    add_dsvector(c->im, t3, t4);

    free_dsvector(t1);
    free_dsvector(t2);
    free_dsvector(t3);
    free_dsvector(t4);
}

/* c = val * a */
void cmul_cdsvector_3m(CDSVector c, cdsfloat *val, CDSVector a)
{
    DSVector t1, t2, t3;
    dsfloat tmp;

    t1 = init_dsvector(c->re->dim);
    t2 = init_dsvector(c->re->dim);

    cmul_dsvector(t1, val->val_re, a->re);
    cmul_dsvector(t2, val->val_im, a->im);
    sub_dsvector(c->re, t1, t2);

    //#ifdef USE_4M
        // 4M
    /*
        cmul_dsvector(t1, val->val_im, a->re);
        cmul_dsvector(t2, val->val_re, a->im);
        add_dsvector(c->im, t1, t2);
    */
    //#else // USE_4M
        // 3M
        rds_add(tmp.val, val->val_re, val->val_im);
        t3 = init_dsvector(c->re->dim);
        add_dsvector(t3, a->re, a->im);
        cmul_dsvector(c->im, tmp.val, t3);
        sub_dsvector(c->im, c->im, t1);
        sub_dsvector(c->im, c->im, t2);
        free_dsvector(t3);
    //#endif // USE_4M

    free_dsvector(t1);
    free_dsvector(t2);
}


/* c *= val */
void cmul2_cdsvector(CDSVector c, cdsfloat *val)
{
    CDSVector in_a;

    in_a = init_cdsvector(c->re->dim);

    subst_cdsvector(in_a, c);
    cmul_cdsvector(c, val, in_a);

    free_cdsvector(in_a);
}

/* c = a + val * b */
void add_cmul_cdsvector(CDSVector c, CDSVector a, cdsfloat *val, CDSVector b)
{
    CDSVector in_vec;
    in_vec = init_cdsvector(b->re->dim);

    //cmul_cdsvector(c, val, b);
    cmul_cdsvector(in_vec, val, b);
    //add2_cdsvector(c, a);
    add_cdsvector(c, a, in_vec);

    free_cdsvector(in_vec);
}

/* c = a - val * b */
void sub_cmul_cdsvector(CDSVector c, CDSVector a, cdsfloat *val, CDSVector b)
{
    CDSVector in_vec;
    in_vec = init_cdsvector(b->re->dim);

    //cmul_cdsvector(c, val, b);
    cmul_cdsvector(in_vec, val, b);
    //sub2_cdsvector(c, a);
    sub_cdsvector(c, a, in_vec);

    free_cdsvector(in_vec);
}

/* (a, b) = conj(a)^T * b */
void ip_cdsvector(cdsfloat *ret, CDSVector a, CDSVector b)
{
    int i;
    cdsfloat tmp, conj_a_i, ai, bi;

    rcds_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cdsvector_i(&ai, a, i);
        rcds_conj(&conj_a_i, &ai); // get_cdsvector_i(a, i));
        subst_cdsvector_i(&bi, b, i);
        rcds_mul(&tmp, &conj_a_i, &bi); // get_cdsvector_i(b, i));
        rcds_add(ret, ret, &tmp);
    }
}

/* a^T * b */
void dotp_cdsvector(cdsfloat *ret, CDSVector a, CDSVector b)
{
    int i;
    cdsfloat tmp, ai, bi;

    rcds_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cdsvector_i(&ai, a, i);
        subst_cdsvector_i(&bi, b, i);
        rcds_mul(&tmp, &ai, &bi); // get_cdsvector_i(a, i), get_cdsvector_i(b, i));
        rcds_add(ret, ret, &tmp);
    }
}

/* c := a */
void subst_cdsvector(CDSVector c, CDSVector a)
{
    subst_dsvector(c->re, a->re);
    subst_dsvector(c->im, a->im);
}

/* c := conj(a) */
void conj_cdsvector(CDSVector c, CDSVector a)
{
    subst_dsvector(c->re, a->re);
    neg_dsvector(c->im, a->im);
}

/* c := -a */
void neg_cdsvector(CDSVector c, CDSVector a)
{
    neg_dsvector(c->re, a->re);
    neg_dsvector(c->im, a->im);
}

/* ||a||_1 */
void norm1_cdsvector(float ret[DSSIZE], CDSVector a)
{
    int i;
    dsfloat tmp;
    cdsfloat ai;

    rds_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cdsvector_i(&ai, a, i);
        rcds_abs(&tmp, &ai); // get_cdsvector_i(a, i));
        rds_add(ret, ret, tmp.val);
    }
}

/* ||a||_infty */
void normi_cdsvector(float ret[DSSIZE], CDSVector a)
{
    int i;
    dsfloat tmp;
    cdsfloat ai;

    rds_set0(ret);
    for(i = 0; i < a->re->dim; i++)
    {
        subst_cdsvector_i(&ai, a, i);
        rcds_abs(&tmp, &ai); // get_cdsvector_i(a, i));
        if(rds_cmp(ret, tmp.val) < 0)
            rds_set(ret, tmp.val);
    }
}

// Euclid norm
void norm2_cdsvector(float ret[DSSIZE], CDSVector vec)
{
    int i;
    dsfloat tmp;
    cdsfloat vec_i;

    rds_set0(ret);
    for(i = 0; i < vec->re->dim; i++)
    {
        subst_cdsvector_i(&vec_i, vec, i);
        rcds_nrm2(&tmp, &vec_i); // get_cdsvector_i(vec, i));
        rds_add(ret, ret, tmp.val);
    }
    rds_sqrt(ret, ret);
}

// set a zero matrix
//void set0_cdsmatrix(CDSMatrix mat);
void set0_cdsmatrix(CDSMatrix mat)
{
    set0_dsmatrix(mat->re);
    set0_dsmatrix(mat->im);
}

// initialize dsvector
CDSMatrix init_cdsmatrix(long int row_dim, long int col_dim)
{
    CDSMatrix ret = NULL;

    ret = (CDSMatrix)malloc(sizeof(cdsmatrix));
    if(ret == NULL)
    {
        fprintf(stderr, "ERROR: init_cdsmatrix(%ld, %ld)\n", row_dim, col_dim);
        return NULL;
    }

    ret->re = init_dsmatrix(row_dim, col_dim);
    if(ret->re == NULL)
    {
        fprintf(stderr, "ERROR: init_cdsmatrix(%ld, %ld)\n", row_dim, col_dim);
        free(ret);
        return NULL;
    }

    ret->im = init_dsmatrix(row_dim, col_dim);
    if(ret->im == NULL)
    {
        fprintf(stderr, "ERROR: init_cdsmatrix(%ld, %ld)\n", row_dim, col_dim);
        free_dsmatrix(ret->re);
        free(ret);
        return NULL;
    }

    return ret;
}

// free csdsvector
void free_cdsmatrix(CDSMatrix mat)
{
    free_dsmatrix(mat->re);
    free_dsmatrix(mat->im);
    free(mat);
}

// print cdsvector
//void print_cdsmatrix(CDSMatrix mat);

// CDSMatrix mat -> cdsfloat array
void set_cdsfloat_cdsmat(cdsfloat ret[], int ret_dim, CDSMatrix mat)
{
    long int i, j, index;
    cdsfloat mat_ij;

    index = 0;
    for(i = 0; i < mat->re->row_dim; i++)
    {
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_cdsmatrix_ij(&mat_ij, mat, i, j);
            rcds_set(&ret[index++], &mat_ij); // get_cdsmatrix_ij(mat, i, j));
        }
    }

    return;
}

// dsmatrix -> cdsmatrix
void set_cdsmatrix_dsmat(CDSMatrix ret, DSMatrix re_mat, DSMatrix im_mat)
{
    subst_dsmatrix(ret->re, re_mat);
    subst_dsmatrix(ret->im, im_mat);
}

// cdsfloat array -> CDDmatrix ret
void set_cdsmatrix_cdsfloat(CDSMatrix ret, cdsfloat array[], int array_dim)
{
    long int i, j, index;

    index = 0;
    for(i = 0; i < ret->re->row_dim; i++)
    {
        for(j = 0; j < ret->re->col_dim; j++)
            set_cdsmatrix_ij(ret, i, j, &array[index++]);
    }

    return;
}

// matrix multiplication
// ret := A * B
void mul_cdsmatrix_4m(CDSMatrix ret, CDSMatrix a, CDSMatrix b)
{
    DSMatrix t1, t2, t3, t4;

    t1 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_dsmatrix(t1, a->re, b->re);
    mul_dsmatrix(t2, a->im, b->im);
    sub_dsmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_dsmatrix(t3, a->im, b->re);
        mul_dsmatrix(t4, a->re, b->im);
        add_dsmatrix(ret->im, t3, t4);
    //#else // USE_4M 
        // 3M
    /*
        add_dsmatrix(t3, a->re, a->im);
        add_dsmatrix(t4, b->re, b->im);
        mul_dsmatrix(ret->im, t3, t4);
        sub_dsmatrix(ret->im, ret->im, t1);
        sub_dsmatrix(ret->im, ret->im, t2);
    */
    //#endif // USE_4M

    free_dsmatrix(t1);
    free_dsmatrix(t2);
    free_dsmatrix(t3);
    free_dsmatrix(t4);
}

// matrix multiplication
// ret := A * B
void mul_cdsmatrix_3m(CDSMatrix ret, CDSMatrix a, CDSMatrix b)
{
    DSMatrix t1, t2, t3, t4;

    t1 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_dsmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_dsmatrix(b->re->row_dim, ret->re->col_dim);

    mul_dsmatrix(t1, a->re, b->re);
    mul_dsmatrix(t2, a->im, b->im);
    sub_dsmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
    /*
        mul_dsmatrix(t3, a->im, b->re);
        mul_dsmatrix(t4, a->re, b->im);
        add_dsmatrix(ret->im, t1, t2);
    */
    //#else // USE_4M 
        // 3M
        add_dsmatrix(t3, a->re, a->im);
        add_dsmatrix(t4, b->re, b->im);
        mul_dsmatrix(ret->im, t3, t4);
        sub_dsmatrix(ret->im, ret->im, t1);
        sub_dsmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_dsmatrix(t1);
    free_dsmatrix(t2);
    free_dsmatrix(t3);
    free_dsmatrix(t4);
}


// Frobenius norm
void normf_cdsmatrix(float ret[DSSIZE], CDSMatrix mat)
{
    int i, j;
    dsfloat tmp;
    cdsfloat mat_ij;

    rds_set0(ret);
    for(i = 0; i < mat->re->row_dim; i++)
    {
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_cdsmatrix_ij(&mat_ij, mat, i, j);
            rcds_nrm2(&tmp, &mat_ij); // get_cdsmatrix_ij(mat, i, j));
            rds_add(ret, ret, tmp.val);
        }
    }
    rds_sqrt(ret, ret);
}

// print normf
void print_normf_cdsmatrix(const char *str, CDSMatrix mat)
{
    dsfloat tmp;

    normf_cdsmatrix(tmp.val, mat);

    rds_out_str_base(stdout, 10, 33, tmp.val);
}

/*************************************************/
/* Matrix Caluculations for CDSMatrix            */
/*
void normf_cdsmatrix(float ret[DSSIZE], CDSMatrix mat)
void norm1_cdsmatrix(float ret[DSSIZE], CDSMatrix mat)
void normi_cdsmatrix(float ret[DSSIZE], CDSMatrix mat)
void add_cdsmatrix(CDSMatrix c, CDSMatrix a, CDSMatrix b);
void sub_cdsmatrix(CDSMatrix c, CDSMatrix a, CDSMatrix b);
void mul_cdsmatrix(CDSMatrix c, CDSMatrix a, CDSMatrix b);
void mul_cdsmatrix_dsvec(CDSVector v, CDSMatrix a, CDSVector vb)
void mul_cdsmatrixt_dsvec(CDSVector v, CDSMatrix a, CDSVector vb)
void transpose_cdsmatrix(CDSMatrix c, CDSMatrix a);
void inv_cdsmatrix(CDSMatrix a);
void subst_mpfmatrux(CDSMatrix c, CDSMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_cdsmatrix(float ret[DSSIZE], CDSMatrix mat)
{
    int i, j;
    dsfloat tmp, sum;
    cdsfloat mat_ij;

    rds_set0(ret);
    for(j = 0; j < mat->re->col_dim; j++)
    {
        rds_set0(sum.val);
        for(i = 0; i < mat->re->row_dim; i++)
        {
            subst_cdsmatrix_ij(&mat_ij, mat, i, j);
            rcds_abs(&tmp, &mat_ij); // get_cdsmatrix_ij(mat, i, j));
            rds_add(sum.val, sum.val, tmp.val);
        }
        if(rds_cmp(ret, sum.val) < 0)
            rds_set(ret, sum.val);
    }
}

/* 1 Norm of Matrix */
void norm1_cdsmatrix(float ret[DSSIZE], CDSMatrix mat)
{
    int i, j;
    dsfloat tmp, sum;
    cdsfloat mat_ij;

    rds_set0(ret);
    for(i = 0; i < mat->re->row_dim; i++)
    {
        rds_set0(sum.val);
        for(j = 0; j < mat->re->col_dim; j++)
        {
            subst_cdsmatrix_ij(&mat_ij, mat, i, j);
            rcds_abs(&tmp, &mat_ij); // get_cdsmatrix_ij(mat, i, j));
            rds_add(sum.val, sum.val, tmp.val);
        }
        if(rds_cmp(ret, sum.val) < 0)
            rds_set(ret, sum.val);
    }
}

/* c := a + b */
void add_cdsmatrix(CDSMatrix c, CDSMatrix a, CDSMatrix b)
{
    add_dsmatrix(c->re, a->re, b->re);
    add_dsmatrix(c->im, a->im, b->im);
}

/* c := a - b */
void sub_cdsmatrix(CDSMatrix c, CDSMatrix a, CDSMatrix b)
{
    sub_dsmatrix(c->re, a->re, b->re);
    sub_dsmatrix(c->im, a->im, b->im);
}

/* c := sc * a */
void cmul_cdsmatrix(CDSMatrix c, cdsfloat *sc, CDSMatrix a)
{
    DSMatrix t1, t2, t3;
    dsfloat tmp;

    t1 = init_dsmatrix(c->re->row_dim, c->re->col_dim);
    t2 = init_dsmatrix(c->re->row_dim, c->re->col_dim);

    cmul_dsmatrix(t1, sc->val_re, a->re);
    cmul_dsmatrix(t2, sc->val_im, a->im);
    sub_dsmatrix(c->re, t1, t2);

    #ifdef USE_4M
        // 4M
        cmul_dsmatrix(t1, sc->val_im, a->re);
        cmul_dsmatrix(t2, sc->val_re, a->im);
        add_dsmatrix(c->im, t1, t2);
    #else // USE_4M
        // 3M
        rds_add(tmp.val, sc->val_re, sc->val_im);
        t3 = init_dsmatrix(c->re->row_dim, c->im->col_dim);
        add_dsmatrix(t3, a->re, a->im);
        cmul_dsmatrix(c->im, tmp.val, t3);
        sub_dsmatrix(c->im, c->im, t1);
        sub_dsmatrix(c->im, c->im, t2);
        free_dsmatrix(t3);
    #endif // USE_4M

    free_dsmatrix(t1);
    free_dsmatrix(t2);
}


/* c = a^T */
void transpose_cdsmatrix(CDSMatrix c, CDSMatrix a)
{
    transpose_dsmatrix(c->re, a->re);
    transpose_dsmatrix(c->im, a->im);
}

/* c := conj(a)^T */
void star_cdsmatrix(CDSMatrix c, CDSMatrix a)
{
    // c_re := -a_im
    _cds_neg_dsmatrix(c->re, a->im);
    // c_im := -a_im^T
    transpose_dsmatrix(c->im, c->re);
    // c_re := a_re^T
    transpose_dsmatrix(c->re, a->re);
}

/* c := a */
void subst_cdsmatrix(CDSMatrix c, CDSMatrix a)
{
    subst_dsmatrix(c->re, a->re);
    subst_dsmatrix(c->im, a->im);
}

/* c := conj(a) */
void conj_cdsmatrix(CDSMatrix c, CDSMatrix a)
{
    subst_dsmatrix(c->re, a->re);
    _cds_neg_dsmatrix(c->im, a->im);
}

/* c := -a */
void neg_cdsmatrix(CDSMatrix c, CDSMatrix a)
{
    _cds_neg_dsmatrix(c->re, a->re);
    _cds_neg_dsmatrix(c->im, a->im);
}

/* c := I */
void setI_cdsmatrix(CDSMatrix c)
{
    setI_dsmatrix(c->re);
    set0_dsmatrix(c->im);
}

/* v := a * vb */
void mul_cdsmatrix_cdsvec_4m(CDSVector v, CDSMatrix a, CDSVector vb)
{
    DSVector t1, t2, t3, t4;
    DSMatrix tmp_mat;

    t1 = init_dsvector(v->re->dim);
    t2 = init_dsvector(v->re->dim);
    t3 = init_dsvector(v->re->dim);
    t4 = init_dsvector(v->re->dim);

    mul_dsmatrix_dsvec(t1, a->re, vb->re);
    mul_dsmatrix_dsvec(t2, a->im, vb->im);
    sub_dsvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        mul_dsmatrix_dsvec(t3, a->im, vb->re);
        mul_dsmatrix_dsvec(t4, a->re, vb->im);
        add_dsvector(v->im, t3, t4);
    //#else // USE_4M
    /*
        // 3M
        tmp_mat = init_dsmatrix(a->re->row_dim, a->re->col_dim);
        add_dsmatrix(tmp_mat, a->re, a->im);
        add_dsvector(t3, vb->re, vb->im);
        mul_dsmatrix_dsvec(t4, tmp_mat, t3);
        sub_dsvector(v->im, t4, t1);
        sub_dsvector(v->im, v->im, t2);
        free_dsmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_dsvector(t1);
    free_dsvector(t2);
    free_dsvector(t3);
    free_dsvector(t4);
}

/* v := a * vb */
void mul_cdsmatrix_cdsvec_3m(CDSVector v, CDSMatrix a, CDSVector vb)
{
    DSVector t1, t2, t3, t4;
    DSMatrix tmp_mat;

    t1 = init_dsvector(v->re->dim);
    t2 = init_dsvector(v->re->dim);
    t3 = init_dsvector(v->re->dim);
    t4 = init_dsvector(v->re->dim);

    mul_dsmatrix_dsvec(t1, a->re, vb->re);
    mul_dsmatrix_dsvec(t2, a->im, vb->im);
    sub_dsvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
    /*
        mul_dsmatrix_dsvec(t3, a->im, vb->re);
        mul_dsmatrix_dsvec(t4, a->re, vb->im);
        add_dsvector(v->im, t3, t4);
    */
    //#else // USE_4M
        // 3M
        tmp_mat = init_dsmatrix(a->re->row_dim, a->re->col_dim);
        add_dsmatrix(tmp_mat, a->re, a->im);
        add_dsvector(t3, vb->re, vb->im);
        mul_dsmatrix_dsvec(t4, tmp_mat, t3);
        sub_dsvector(v->im, t4, t1);
        sub_dsvector(v->im, v->im, t2);
        free_dsmatrix(tmp_mat);
    //#endif // USE_4M

    free_dsvector(t1);
    free_dsvector(t2);
    free_dsvector(t3);
    free_dsvector(t4);
}


/* v := a^T * vb */
void mul_cdsmatrixt_cdsvec(CDSVector v, CDSMatrix a, CDSVector vb)
{
    DSVector t1, t2, t3, t4;
    DSMatrix tmp_mat;

    t1 = init_dsvector(v->re->dim);
    t2 = init_dsvector(v->re->dim);
    t3 = init_dsvector(v->re->dim);
    t4 = init_dsvector(v->re->dim);

    mul_dsmatrixt_dsvec(t1, a->re, vb->re);
    mul_dsmatrixt_dsvec(t2, a->im, vb->im);
    sub_dsvector(v->re, t1, t2);

   // #ifdef USE_4M
        // 4M
        mul_dsmatrixt_dsvec(t3, a->im, vb->re);
        mul_dsmatrixt_dsvec(t4, a->re, vb->im);
        add_dsvector(v->im, t3, t4);
    //#else // USE_4M
    /*
        // 3M
        tmp_mat = init_dsmatrix(a->re->row_dim, a->re->col_dim);
        add_dsmatrix(tmp_mat, a->re, a->im);
        add_dsvector(t3, vb->re, vb->im);
        mul_dsmatrixt_dsvec(t4, tmp_mat, t3);
        sub_dsvector(v->im, t4, t1);
        sub_dsvector(v->im, v->im, t2);
        free_dsmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_dsvector(t1);
    free_dsvector(t2);
    free_dsvector(t3);
    free_dsvector(t4);
}

/* v := conj(a)^T * vb */
void mul_cdsmatrixs_cdsvec(CDSVector v, CDSMatrix a, CDSVector vb)
{
    DSVector t1, t2, t3, t4;
    DSMatrix tmp_mat;

    t1 = init_dsvector(v->re->dim);
    t2 = init_dsvector(v->re->dim);
    t3 = init_dsvector(v->re->dim);
    t4 = init_dsvector(v->re->dim);

    mul_dsmatrixt_dsvec(t1, a->re, vb->re);
    mul_dsmatrixt_dsvec(t2, a->im, vb->im);
    add_dsvector(v->re, t1, t2);

   // #ifdef USE_4M
        // 4M
        mul_dsmatrixt_dsvec(t3, a->im, vb->re);
        mul_dsmatrixt_dsvec(t4, a->re, vb->im);
        sub_dsvector(v->im, t3, t4);
    //#else // USE_4M
    /*
        // 3M
        tmp_mat = init_dsmatrix(a->re->row_dim, a->re->col_dim);
        add_dsmatrix(tmp_mat, a->re, a->im);
        add_dsvector(t3, vb->re, vb->im);
        mul_dsmatrixt_dsvec(t4, tmp_mat, t3);
        sub_dsvector(v->im, t4, t1);
        sub_dsvector(v->im, v->im, t2);
        free_dsmatrix(tmp_mat);
    */
    //#endif // USE_4M

    free_dsvector(t1);
    free_dsvector(t2);
    free_dsvector(t3);
    free_dsvector(t4);
}


/* a = a^(-1) */
/* square matrix only */
void inv_cdsmatrix(CDSMatrix a)
{
	long int i, j, k, dim;
	cdsfloat ctmp, aii, aij, aik, ajk, aji;
	dsfloat tmp;

	/* Check Dimensions */
	if(a->re->row_dim != a->re->col_dim)
	{
		fprintf(stderr, "ERROR: inv_cdsmatrix\n");
		return;
	}

	dim = a->re->row_dim;

	for(i = 0; i < dim; i++)
	{
		rcds_abs(&tmp, get_cdsmatrix_ij(a, i, i));
		if(rds_cmp_ui(tmp.val, 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_cdsmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

        subst_cdsmatrix_ij(&ctmp, a, i, i);
		rcds_inv(&aii, &ctmp); // get_cdsmatrix_ij(a, i, i));
		//inv_mpc_t(aii, get_cmpfmatrix_ij(a, i, i));
		set_cdsmatrix_ij(a, i, i, &aii);

		for(j = 0; j < i; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
            subst_cdsmatrix_ij(&aij, a, i, j);
			rcds_mul(&ctmp, &aij, &aii); // get_cdsmatrix_ij(a, i, j), &aii);
			set_cdsmatrix_ij(a, i, j, &ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
            subst_cdsmatrix_ij(&aij, a, i, j);
			rcds_mul(&ctmp, &aij, &aii); // get_cdsmatrix_ij(a, i, j), &aii);
			set_cdsmatrix_ij(a, i, j, &ctmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cdsmatrix_ij(&aji, a, j, i);
                subst_cdsmatrix_ij(&aik, a, i, k);
                subst_cdsmatrix_ij(&ajk, a, j, k);
				rcds_mul(&ctmp, &aji, &aik); // get_cdsmatrix_ij(a, j, i), get_cdsmatrix_ij(a, i, k));
				rcds_sub(&ctmp, &ajk, &ctmp); // get_cdsmatrix_ij(a, j, k), &ctmp);
				set_cdsmatrix_ij(a, j, k, &ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cdsmatrix_ij(&aji, a, j, i);
                subst_cdsmatrix_ij(&aik, a, i, k);
                subst_cdsmatrix_ij(&ajk, a, j, k);
				rcds_mul(&ctmp, &aji, &aik); // get_cdsmatrix_ij(a, j, i), get_cdsmatrix_ij(a, i, k));
				rcds_sub(&ctmp, &ajk, &ctmp); // get_cdsmatrix_ij(a, j, k), &ctmp);
				set_cdsmatrix_ij(a, j, k, &ctmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cdsmatrix_ij(&aji, a, j, i);
                subst_cdsmatrix_ij(&aik, a, i, k);
                subst_cdsmatrix_ij(&ajk, a, j, k);
				rcds_mul(&ctmp, &aji, &aik); // get_cdsmatrix_ij(a, j, i), get_cdsmatrix_ij(a, i, k));
				rcds_sub(&ctmp, &ajk, &ctmp); // get_cdsmatrix_ij(a, j, k), &ctmp);
				set_cdsmatrix_ij(a, j, k, &ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
                subst_cdsmatrix_ij(&aji, a, j, i);
                subst_cdsmatrix_ij(&aik, a, i, k);
                subst_cdsmatrix_ij(&ajk, a, j, k);
				rcds_mul(&ctmp, &aji, &aik); // get_cdsmatrix_ij(a, j, i), get_cdsmatrix_ij(a, i, k));
				rcds_sub(&ctmp, &ajk, &ctmp); // get_cdsmatrix_ij(a, j, k), &ctmp);
				set_cdsmatrix_ij(a, j, k, &ctmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			rcds_neg(&ctmp, &aii);
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
            subst_cdsmatrix_ij(&aji, a, j, i);
			rcds_mul(&ctmp, &ctmp, &aji); // get_cdsmatrix_ij(a, j, i));
			set_cdsmatrix_ij(a, j, i, &ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			rcds_neg(&ctmp, &aii);
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
            subst_cdsmatrix_ij(&aji, a, j, i);
			rcds_mul(&ctmp, &ctmp, &aji); // get_cdsmatrix_ij(a, j, i));
			set_cdsmatrix_ij(a, j, i, &ctmp);
		}
	}
}

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpc)a */
void subst_cmpfvector_cdsvec(CMPFVector c, CDSVector a)
{
    MPFVector re, im;

    re = init2_mpfvector(c->dim, c->prec);
    im = init2_mpfvector(c->dim, c->prec);

    subst_mpfvector_dsvec(re, a->re);
    subst_mpfvector_dsvec(im, a->im);

    merge_cmpfvector(c, re, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (dd)a */
void subst_cdsvector_cmpfvec(CDSVector c, CMPFVector a)
{
    MPFVector re, im;

    re = init2_mpfvector(a->dim, a->prec);
    im = init2_mpfvector(a->dim, a->prec);

    separate_cmpfvector(re, im, a);
    subst_dsvector_mpfvec(c->re, re);
    subst_dsvector_mpfvec(c->im, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (dd)a / max|a_i| */
void subst_coef_cdsvector_cmpfvec(CDSVector c, CMPFVector a)
{
    MPFVector re, im;
    mpc_t absmax_a;

    re = init2_mpfvector(a->dim, a->prec);
    im = init2_mpfvector(a->dim, a->prec);

    separate_cmpfvector(re, im, a);
    subst_dsvector_mpfvec(c->re, re);
    subst_dsvector_mpfvec(c->im, im);

    free_mpfvector(re);
    free_mpfvector(im);
}

/* c := (mpf)a */
void subst_cmpfmatrix_cdsmat(CMPFMatrix c, CDSMatrix a)
{
    MPFMatrix re, im;

    re = init2_mpfmatrix(c->row_dim, c->col_dim, c->prec);
    im = init2_mpfmatrix(c->row_dim, c->col_dim, c->prec);

    subst_mpfmatrix_dsmat(re, a->re);
    subst_mpfmatrix_dsmat(im, a->im);

    merge_cmpfmatrix(c, re, im);

    free_mpfmatrix(re);
    free_mpfmatrix(im);
}


/* c := (dd)a */
void subst_cdsmatrix_cmpfmat(CDSMatrix c, CMPFMatrix a)
{
    MPFMatrix re, im;

    re = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    im = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);

    separate_cmpfmatrix(re, im, a);
    subst_dsmatrix_mpfmat(c->re, re);
    subst_dsmatrix_mpfmat(c->im, im);

    free_mpfmatrix(re);
    free_mpfmatrix(im);
}

/* Normwise relative error of vector */
void relerr_cdsvector_cmpfvec(float relerr[DSSIZE], CDSVector approx_vec, CMPFVector true_vec, int norm_type)
{}

/* Elementwise relative errors of vector */
void relerr_element_cdsvector_mpf(float max_relerr[DSSIZE], float min_relerr[DSSIZE], float norm_relerr[DSSIZE], CDSVector approx_vec, MPFVector true_vec, int norm_type)
{}
#endif // USE_GMP

/* Normwise relative error of vector */
void relerr_cdsvector(float relerr[DSSIZE], CDSVector approx_vec, CDSVector true_vec, int norm_type)
{
    float norm_true_vec[DSSIZE], norm_diff_vec[DSSIZE];
	CDSVector diff_vec;

	diff_vec = init_cdsvector(approx_vec->re->dim);

	// diff_vec := approx_vec - true_vec
	sub_cdsvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_cdsvector(norm_diff_vec, diff_vec);
			normi_cdsvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_cdsvector(norm_diff_vec, diff_vec);
			norm1_cdsvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_cdsvector(norm_diff_vec, diff_vec);
			norm2_cdsvector(norm_true_vec, true_vec);
			break;
	}

	if(rds_cmp_ui(norm_true_vec, 0UL) != 0)
		rds_div(relerr, norm_diff_vec, norm_true_vec);

	free_cdsvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_cdsvector(float max_relerr[DSSIZE], float min_relerr[DSSIZE], float norm_relerr[DSSIZE], CDSVector approx_vec, CDSVector true_vec, int norm_type)
{
    dsfloat abs_true_vec, abs_diff_vec;
    float norm_diff_vec[DSSIZE], norm_true_vec[DSSIZE];
	long int i;
	CDSVector diff_vec;
    cdsfloat ctmp;

	diff_vec = init_cdsvector(approx_vec->re->dim);

	// diff_vec := approx_vec - true_vec
	sub_cdsvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_cdsvector(norm_diff_vec, diff_vec);
			normi_cdsvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_cdsvector(norm_diff_vec, diff_vec);
			norm1_cdsvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_cdsvector(norm_diff_vec, diff_vec);
			norm2_cdsvector(norm_true_vec, true_vec);
			break;
	}

	rds_set(norm_relerr, norm_diff_vec);
	if(rds_cmp_ui(norm_true_vec, 0UL) != 0)
		rds_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rds_set_ui(max_relerr, 0UL);
	normi_cdsvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->re->dim; i++)
	{
        subst_cdsvector_i(&ctmp, diff_vec, i);
		rcds_abs(&abs_diff_vec, &ctmp); // get_cdsvector_i(diff_vec, i));
        subst_cdsvector_i(&ctmp, true_vec, i);
		rcds_abs(&abs_true_vec, &ctmp); // get_cdsvector_i(true_vec, i));
		if(rds_cmp_ui(abs_true_vec.val, 0UL) != 0)
			rds_div(abs_diff_vec.val, abs_diff_vec.val, abs_true_vec.val);
		
		if(rds_cmp(max_relerr, abs_diff_vec.val) < 0)
			rds_set(max_relerr, abs_diff_vec.val);
		if(rds_cmp(min_relerr, abs_diff_vec.val) > 0)
			rds_set(min_relerr, abs_diff_vec.val);
	}

	free_cdsvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_cdsmatrix(CDSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
    row_swap_dsmatrix(mat->re, row_index0, row_index1, col_start, col_end);
    row_swap_dsmatrix(mat->im, row_index0, row_index1, col_start, col_end);
}

// print cdsmatrix
void print_cdsmatrix(CDSMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->re->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->re->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_dd_write((vec->element + index * DSSIZE));
			//c_dd_write(GET_DDMATRIX_IJ(mat, row_index, col_index));
			rds_out_str_base(stdout, 10, 33, get_cdsmatrix_ij_cdsfloat(mat, row_index, col_index).val_re);
            printf(" + ");
   			rds_out_str_base(stdout, 10, 33, get_cdsmatrix_ij_cdsfloat(mat, row_index, col_index).val_im);
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
    dsfloat a, b, c
    cdsfloat ca, cb, cc;

    rds_set_d(a->val, 3.0); rds_sqrt(a->val);
    rds_set_d(b->val, 5.0); rds_sqrt(b->val);

    rds_out_str_base(stdout, 10, 33, a);
    rds_out_str_base(stdout, 10, 33, b);
    
}
#endif // DEBUG

