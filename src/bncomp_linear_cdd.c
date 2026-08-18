/********************************************************************************/
/* bncomp_linear_cdd.c: Parallelized Complex DD Precision                       */
/*                                       Linear Computation Library with OpenMP */
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
// BNCpack with OpenMP
#include "bncomp.h"
#include "matmul_strassen.h"

//---------------------------------------
// DD
//---------------------------------------

//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_cddvector(CDDVector c, CDDVector a)
{
    _bncomp_subst_ddvector(c->re, a->re);
    _bncomp_subst_ddvector(c->im, a->im);
}

/* c = a + b */
void _bncomp_add_cddvector(CDDVector c, CDDVector a, CDDVector b)
{
    _bncomp_add_ddvector(c->re, a->re, b->re);
    _bncomp_add_ddvector(c->im, a->im, b->im);
}

/* c = a - b */
void _bncomp_sub_cddvector(CDDVector c, CDDVector a, CDDVector b)
{
    _bncomp_sub_ddvector(c->re, a->re, b->re);
    _bncomp_sub_ddvector(c->im, a->im, b->im);
}

/* c = val * a */
void _bncomp_cmul_cddvector_4m(CDDVector c, cddfloat *val, CDDVector a)
{
    DDVector t1, t2, t3;
    ddfloat tmp;

    t1 = init_ddvector(c->re->dim);
    t2 = init_ddvector(c->re->dim);

    _bncomp_cmul_ddvector(t1, val->val_re, a->re);
    _bncomp_cmul_ddvector(t2, val->val_im, a->im);
    _bncomp_sub_ddvector(c->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_cmul_ddvector(t1, val->val_im, a->re);
        _bncomp_cmul_ddvector(t2, val->val_re, a->im);
        _bncomp_add_ddvector(c->im, t1, t2);
    //#else // USE_4M
        // 3M
    /*
        rdd_add(tmp.val, val->val_re, val->val_im);
        t3 = init_ddvector(c->re->dim);
        add_ddvector(t3, a->re, a->im);
        cmul_ddvector(c->im, tmp.val, t3);
        sub_ddvector(c->im, c->im, t1);
        sub_ddvector(c->im, c->im, t2);
        free_ddvector(t3);
    */
    //#endif // USE_4M

    free_ddvector(t1);
    free_ddvector(t2);
}

/* c = val * a */
void _bncomp_cmul_cddvector_3m(CDDVector c, cddfloat *val, CDDVector a)
{
    DDVector t1, t2, t3;
    ddfloat tmp;

    t1 = init_ddvector(c->re->dim);
    t2 = init_ddvector(c->re->dim);

    _bncomp_cmul_ddvector(t1, val->val_re, a->re);
    _bncomp_cmul_ddvector(t2, val->val_im, a->im);
    _bncomp_sub_ddvector(c->re, t1, t2);

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
        _bncomp_add_ddvector(t3, a->re, a->im);
        _bncomp_cmul_ddvector(c->im, tmp.val, t3);
        _bncomp_sub_ddvector(c->im, c->im, t1);
        _bncomp_sub_ddvector(c->im, c->im, t2);
        free_ddvector(t3);
    //#endif // USE_4M

    free_ddvector(t1);
    free_ddvector(t2);
}

/* (a, b) */
/* (a, b) = conj(a)^T * b */
void _bncomp_ip_cddvector(cddfloat *ret, CDDVector a, CDDVector b)
{
    int thread_index;
    long int i;
    cddfloat tmp[BNCOMP_MAX_NUM_THREADS], conj_a_i[BNCOMP_MAX_NUM_THREADS];

    rcdd_set0(ret);

    #pragma omp parallel for private(thread_index)
    for(i = 0; i < a->re->dim; i++)
    {
		thread_index = omp_get_thread_num();

        rcdd_conj(&conj_a_i[thread_index], get_cddvector_i(a, i));
        rcdd_mul(&tmp[thread_index], &conj_a_i[thread_index], get_cddvector_i(b, i));
    #pragma omp critical 
        rcdd_add(ret, ret, &tmp[thread_index]);
    }
}

/* a^T * b */
void _bncomp_dotp_cddvector(cddfloat *ret, CDDVector a, CDDVector b)
{
    int thread_index;
    long int i;
    cddfloat tmp[BNCOMP_MAX_NUM_THREADS];

    rcdd_set0(ret);

    #pragma omp parallel for private(thread_index)
    for(i = 0; i < a->re->dim; i++)
    {
        rcdd_mul(&tmp[thread_index], get_cddvector_i(a, i), get_cddvector_i(b, i));
    #pragma omp critical 
        rcdd_add(ret, ret, &tmp[thread_index]);
    }
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_cddmatrix(CDDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
    _bncomp_row_swap_ddmatrix(mat->re, row_index0, row_index1, col_start, col_end);
    _bncomp_row_swap_ddmatrix(mat->im, row_index0, row_index1, col_start, col_end);
}

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b)
{
    _bncomp_add_ddmatrix(c->re, a->re, b->re);
    _bncomp_add_ddmatrix(c->im, a->im, b->im);
}

/* c := a - b */
void _bncomp_sub_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b)
{
    _bncomp_sub_ddmatrix(c->re, a->re, b->re);
    _bncomp_sub_ddmatrix(c->im, a->im, b->im);
}

/* c := sc * a */
void _bncomp_cmul_cddmatrix(CDDMatrix c, cddfloat *sc, CDDMatrix a)
{
    DDMatrix t1, t2, t3;
    ddfloat tmp;

    t1 = init_ddmatrix(c->re->row_dim, c->re->col_dim);
    t2 = init_ddmatrix(c->re->row_dim, c->re->col_dim);

    _bncomp_cmul_ddmatrix(t1, sc->val_re, a->re);
    _bncomp_cmul_ddmatrix(t2, sc->val_im, a->im);
    _bncomp_sub_ddmatrix(c->re, t1, t2);

    #ifdef USE_4M
        // 4M
        _bncomp_cmul_ddmatrix(t1, sc->val_im, a->re);
        _bncomp_cmul_ddmatrix(t2, sc->val_re, a->im);
        _bncomp_add_ddmatrix(c->im, t1, t2);
    #else // USE_4M
        // 3M
        rdd_add(tmp.val, sc->val_re, sc->val_im);
        t3 = init_ddmatrix(c->re->row_dim, c->im->col_dim);
        _bncomp_add_ddmatrix(t3, a->re, a->im);
        _bncomp_cmul_ddmatrix(c->im, tmp.val, t3);
        _bncomp_sub_ddmatrix(c->im, c->im, t1);
        _bncomp_sub_ddmatrix(c->im, c->im, t2);
        free_ddmatrix(t3);
    #endif // USE_4M
}

/* c := a */
void _bncomp_subst_cddmatrix(CDDMatrix c, CDDMatrix a)
{
    _bncomp_subst_ddmatrix(c->re, a->re);
    _bncomp_subst_ddmatrix(c->im, a->im);
}

/* c := I */
void _bncomp_setI_cddmatrix(CDDMatrix c)
{
    _bncomp_setI_ddmatrix(c->re);
    _bncomp_set0_ddmatrix(c->im);
}

// set a zero matrix
//void set0_ddmatrix(DDMatrix mat)
void _bncomp_set0_cddmatrix(CDDMatrix mat)
{
    _bncomp_set0_ddmatrix(mat->re);
    _bncomp_set0_ddmatrix(mat->im);
}

/* v := a * vb */
void _bncomp_mul_cddmatrix_cddvec_4m(CDDVector v, CDDMatrix a, CDDVector vb)
{
    DDVector t1, t2, t3, t4;
    DDMatrix tmp_mat;

    t1 = init_ddvector(v->re->dim);
    t2 = init_ddvector(v->re->dim);
    t3 = init_ddvector(v->re->dim);
    t4 = init_ddvector(v->re->dim);

    _bncomp_mul_ddmatrix_ddvec(t1, a->re, vb->re);
    _bncomp_mul_ddmatrix_ddvec(t2, a->im, vb->im);
    _bncomp_sub_ddvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_mul_ddmatrix_ddvec(t3, a->im, vb->re);
        _bncomp_mul_ddmatrix_ddvec(t4, a->re, vb->im);
        _bncomp_add_ddvector(v->im, t3, t4);
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
void _bncomp_mul_cddmatrix_cddvec_3m(CDDVector v, CDDMatrix a, CDDVector vb)
{
    DDVector t1, t2, t3, t4;
    DDMatrix tmp_mat;

    t1 = init_ddvector(v->re->dim);
    t2 = init_ddvector(v->re->dim);
    t3 = init_ddvector(v->re->dim);
    t4 = init_ddvector(v->re->dim);

    _bncomp_mul_ddmatrix_ddvec(t1, a->re, vb->re);
    _bncomp_mul_ddmatrix_ddvec(t2, a->im, vb->im);
    _bncomp_sub_ddvector(v->re, t1, t2);

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
        _bncomp_add_ddmatrix(tmp_mat, a->re, a->im);
        _bncomp_add_ddvector(t3, vb->re, vb->im);
        _bncomp_mul_ddmatrix_ddvec(t4, tmp_mat, t3);
        _bncomp_sub_ddvector(v->im, t4, t1);
        _bncomp_sub_ddvector(v->im, v->im, t2);
        free_ddmatrix(tmp_mat);
    //#endif // USE_4M

    free_ddvector(t1);
    free_ddvector(t2);
    free_ddvector(t3);
    free_ddvector(t4);
}


/* v := a^T * vb */
void _bncomp_mul_cddmatrixt_cddvec_3m(CDDVector v, CDDMatrix a, CDDVector vb)
{
    DDVector t1, t2, t3, t4;
    DDMatrix tmp_mat;

    t1 = init_ddvector(v->re->dim);
    t2 = init_ddvector(v->re->dim);
    t3 = init_ddvector(v->re->dim);
    t4 = init_ddvector(v->re->dim);

    _bncomp_mul_ddmatrixt_ddvec(t1, a->re, vb->re);
    _bncomp_mul_ddmatrixt_ddvec(t2, a->im, vb->im);
    _bncomp_sub_ddvector(v->re, t1, t2);

//    #ifdef USE_4M
        // 4M
        /*
        _bncomp_mul_ddmatrixt_ddvec(t3, a->im, vb->re);
        _bncomp_mul_ddmatrixt_ddvec(t4, a->re, vb->im);
        _bncomp_add_ddvector(v->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        tmp_mat = init_ddmatrix(a->re->row_dim, a->re->col_dim);
        _bncomp_add_ddmatrix(tmp_mat, a->re, a->im);
        _bncomp_add_ddvector(t3, vb->re, vb->im);
        _bncomp_mul_ddmatrixt_ddvec(t4, tmp_mat, t3);
        _bncomp_sub_ddvector(v->im, t4, t1);
        _bncomp_sub_ddvector(v->im, v->im, t2);
        free_ddmatrix(tmp_mat);
    //#endif // USE_4M

    free_ddvector(t1);
    free_ddvector(t2);
    free_ddvector(t3);
    free_ddvector(t4);
}

/* v := a^T * vb */
void _bncomp_mul_cddmatrixt_cddvec_4m(CDDVector v, CDDMatrix a, CDDVector vb)
{
    DDVector t1, t2, t3, t4;
    DDMatrix tmp_mat;

    t1 = init_ddvector(v->re->dim);
    t2 = init_ddvector(v->re->dim);
    t3 = init_ddvector(v->re->dim);
    t4 = init_ddvector(v->re->dim);

    _bncomp_mul_ddmatrixt_ddvec(t1, a->re, vb->re);
    _bncomp_mul_ddmatrixt_ddvec(t2, a->im, vb->im);
    _bncomp_sub_ddvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_mul_ddmatrixt_ddvec(t3, a->im, vb->re);
        _bncomp_mul_ddmatrixt_ddvec(t4, a->re, vb->im);
        _bncomp_add_ddvector(v->im, t3, t4);
/*
    #else // USE_4M
        // 3M
        tmp_mat = init_ddmatrix(a->re->row_dim, a->re->col_dim);
        _bncomp_add_ddmatrix(tmp_mat, a->re, a->im);
        _bncomp_add_ddvector(t3, vb->re, vb->im);
        _bncomp_mul_ddmatrixt_ddvec(t4, tmp_mat, t3);
        _bncomp_sub_ddvector(v->im, t4, t1);
        _bncomp_sub_ddvector(v->im, v->im, t2);
        free_ddmatrix(tmp_mat);
    #endif // USE_4M
*/
    free_ddvector(t1);
    free_ddvector(t2);
    free_ddvector(t3);
    free_ddvector(t4);
}

/* v := conj(a)^T * vb */
void _bncomp_mul_cddmatrixs_cddvec(CDDVector v, CDDMatrix a, CDDVector vb)
{
    DDVector t1, t2, t3, t4;
    DDMatrix tmp_mat;

    t1 = init_ddvector(v->re->dim);
    t2 = init_ddvector(v->re->dim);
    t3 = init_ddvector(v->re->dim);
    t4 = init_ddvector(v->re->dim);

    _bncomp_mul_ddmatrixt_ddvec(t1, a->re, vb->re);
    _bncomp_mul_ddmatrixt_ddvec(t2, a->im, vb->im);
    _bncomp_add_ddvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_mul_ddmatrixt_ddvec(t3, a->im, vb->re);
        _bncomp_mul_ddmatrixt_ddvec(t4, a->re, vb->im);
        _bncomp_sub_ddvector(v->im, t3, t4);
/*
    #else // USE_4M
        // 3M
        tmp_mat = init_ddmatrix(a->re->row_dim, a->re->col_dim);
        _bncomp_add_ddmatrix(tmp_mat, a->re, a->im);
        _bncomp_add_ddvector(t3, vb->re, vb->im);
        _bncomp_mul_ddmatrixt_ddvec(t4, tmp_mat, t3);
        _bncomp_sub_ddvector(v->im, t4, t1);
        _bncomp_sub_ddvector(v->im, v->im, t2);
        free_ddmatrix(tmp_mat);
    #endif // USE_4M
*/
    free_ddvector(t1);
    free_ddvector(t2);
    free_ddvector(t3);
    free_ddvector(t4);
}


// Fit dimension to be multiple of min_dim
void _bncomp_mul_cddmatrix_oz_3m(CDDMatrix ret, CDDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    DDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    /*
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    */
    // ret->row_dim == mat_a->row_dim, ret->col_dim == mat_b->col_dim
    if(\
        (ret->re->row_dim != a->re->row_dim) || \
        (ret->re->col_dim != b->re->col_dim) || \
        (a->re->col_dim != b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cddmatrix_oz_3m\n");
        return;
    }
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(a->re->row_dim, a->re->col_dim);
    t4 = init_ddmatrix(b->re->row_dim, b->re->col_dim);

    _bncomp_mul_ddmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_ddmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_ddmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_ddmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_ddmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        add_ddmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_ddmatrix(t3, a->re, a->im);
        _bncomp_add_ddmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        _bncomp_mul_ddmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t1);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cddmatrix_oz_4m(CDDMatrix ret, CDDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    DDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    /*
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    */
    // ret->row_dim == mat_a->row_dim, ret->col_dim == mat_b->col_dim
    if(\
        (ret->re->row_dim != a->re->row_dim) || \
        (ret->re->col_dim != b->re->col_dim) || \
        (a->re->col_dim != b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cddmatrix_block_3m\n");
        return;
    }
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_ddmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_ddmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_ddmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        _bncomp_mul_ddmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        _bncomp_mul_ddmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        _bncomp_add_ddmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_ddmatrix(t3, a->re, a->im);
        add_ddmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_ddmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_ddmatrix(ret->im, ret->im, t1);
        sub_ddmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (3M)
void _bncomp_mul_cddmatrix_3m(CDDMatrix ret, CDDMatrix a, CDDMatrix b)
{
    DDMatrix t1, t2, t3, t4;

    /*
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_ddmatrix(b->re->row_dim, b->re->col_dim);
    */
    // ret->row_dim == mat_a->row_dim, ret->col_dim == mat_b->col_dim
    if(\
        (ret->re->row_dim != a->re->row_dim) || \
        (ret->re->col_dim != b->re->col_dim) || \
        (a->re->col_dim != b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cddmatrix_3m\n");
        return;
    }
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(a->re->row_dim, a->re->col_dim);
    t4 = init_ddmatrix(b->re->row_dim, b->re->col_dim);

    _bncomp_mul_ddmatrix(t1, a->re, b->re);
    _bncomp_mul_ddmatrix(t2, a->im, b->im);
    _bncomp_sub_ddmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        _bncomp_mul_ddmatrix(t3, a->im, b->re);
        _bncomp_mul_ddmatrix(t4, a->re, b->im);
        _bncomp_add_ddmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_ddmatrix(t3, a->re, a->im);
        _bncomp_add_ddmatrix(t4, b->re, b->im);
        _bncomp_mul_ddmatrix(ret->im, t3, t4);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t1);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (4M)
void _bncomp_mul_cddmatrix_4m(CDDMatrix ret, CDDMatrix a, CDDMatrix b)
{
    DDMatrix t1, t2, t3, t4;
 
    /*
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    */
    // ret->row_dim == mat_a->row_dim, ret->col_dim == mat_b->col_dim
    if(\
        (ret->re->row_dim != a->re->row_dim) || \
        (ret->re->col_dim != b->re->col_dim) || \
        (a->re->col_dim != b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cddmatrix_4m\n");
        return;
    }
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(a->re->row_dim, a->re->col_dim);
    t4 = init_ddmatrix(b->re->row_dim, b->re->col_dim);

    _bncomp_mul_ddmatrix(t1, a->re, b->re);
    _bncomp_mul_ddmatrix(t2, a->im, b->im);
    _bncomp_sub_ddmatrix(ret->re, t1, t2);

    // 4M
    
    //#ifdef USE_4M
        _bncomp_mul_ddmatrix(t3, a->im, b->re);
        _bncomp_mul_ddmatrix(t4, a->re, b->im);
        _bncomp_add_ddmatrix(ret->im, t3, t4);
    //#else // USE_4M
	/*
        // 3M
        _bncomp_add_ddmatrix(t3, a->re, a->im);
        _bncomp_add_ddmatrix(t4, b->re, b->im);
        _bncomp_mul_ddmatrix(ret->im, t3, t4s);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t1);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t2);
	*/
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

