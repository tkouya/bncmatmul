/********************************************************************************/
/* bncomp_linear_ctd.c: Parallelized Complex TD Precision                       */
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
// TD
//---------------------------------------

#ifdef USE_TDLINEAR
//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_ctdvector(CTDVector c, CTDVector a)
{
    _bncomp_subst_tdvector(c->re, a->re);
    _bncomp_subst_tdvector(c->im, a->im);
}

/* c = a + b */
void _bncomp_add_ctdvector(CTDVector c, CTDVector a, CTDVector b)
{
    _bncomp_add_tdvector(c->re, a->re, b->re);
    _bncomp_add_tdvector(c->im, a->im, b->im);
}

/* c = a - b */
void _bncomp_sub_ctdvector(CTDVector c, CTDVector a, CTDVector b)
{
    _bncomp_sub_tdvector(c->re, a->re, b->re);
    _bncomp_sub_tdvector(c->im, a->im, b->im);
}

/* c = val * a */
void _bncomp_cmul_ctdvector_4m(CTDVector c, ctdfloat *val, CTDVector a)
{
    TDVector t1, t2, t3;
    tdfloat tmp;

    t1 = init_tdvector(c->re->dim);
    t2 = init_tdvector(c->re->dim);

    _bncomp_cmul_tdvector(t1, val->val_re, a->re);
    _bncomp_cmul_tdvector(t2, val->val_im, a->im);
    _bncomp_sub_tdvector(c->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_cmul_tdvector(t1, val->val_im, a->re);
        _bncomp_cmul_tdvector(t2, val->val_re, a->im);
        _bncomp_add_tdvector(c->im, t1, t2);
    //#else // USE_4M
        // 3M
    /*
        rtd_add(tmp.val, val->val_re, val->val_im);
        t3 = init_tdvector(c->re->dim);
        add_tdvector(t3, a->re, a->im);
        cmul_tdvector(c->im, tmp.val, t3);
        sub_tdvector(c->im, c->im, t1);
        sub_tdvector(c->im, c->im, t2);
        free_tdvector(t3);
    */
    //#endif // USE_4M

    free_tdvector(t1);
    free_tdvector(t2);
}

/* c = val * a */
void _bncomp_cmul_ctdvector_3m(CTDVector c, ctdfloat *val, CTDVector a)
{
    TDVector t1, t2, t3;
    tdfloat tmp;

    t1 = init_tdvector(c->re->dim);
    t2 = init_tdvector(c->re->dim);

    _bncomp_cmul_tdvector(t1, val->val_re, a->re);
    _bncomp_cmul_tdvector(t2, val->val_im, a->im);
    _bncomp_sub_tdvector(c->re, t1, t2);

    //#ifdef USE_4M
        // 4M
    /*
        cmul_tdvector(t1, val->val_im, a->re);
        cmul_tdvector(t2, val->val_re, a->im);
        add_tdvector(c->im, t1, t2);
    */
    //#else // USE_4M
        // 3M
        rtd_add(tmp.val, val->val_re, val->val_im);
        t3 = init_tdvector(c->re->dim);
        _bncomp_add_tdvector(t3, a->re, a->im);
        _bncomp_cmul_tdvector(c->im, tmp.val, t3);
        _bncomp_sub_tdvector(c->im, c->im, t1);
        _bncomp_sub_tdvector(c->im, c->im, t2);
        free_tdvector(t3);
    //#endif // USE_4M

    free_tdvector(t1);
    free_tdvector(t2);
}

/* (a, b) */
/* (a, b) = conj(a)^T * b */
void _bncomp_ip_ctdvector(ctdfloat *ret, CTDVector a, CTDVector b)
{
    int thread_index;
    long int i;
    ctdfloat tmp[BNCOMP_MAX_NUM_THREADS], conj_a_i[BNCOMP_MAX_NUM_THREADS];

    rctd_set0(ret);

    #pragma omp parallel for private(thread_index)
    for(i = 0; i < a->re->dim; i++)
    {
		thread_index = omp_get_thread_num();

        rctd_conj(&conj_a_i[thread_index], get_ctdvector_i(a, i));
        rctd_mul(&tmp[thread_index], &conj_a_i[thread_index], get_ctdvector_i(b, i));
    #pragma omp critical 
        rctd_add(ret, ret, &tmp[thread_index]);
    }
}

/* a^T * b */
void _bncomp_dotp_ctdvector(ctdfloat *ret, CTDVector a, CTDVector b)
{
    int thread_index;
    long int i;
    ctdfloat tmp[BNCOMP_MAX_NUM_THREADS];

    rctd_set0(ret);

    #pragma omp parallel for private(thread_index)
    for(i = 0; i < a->re->dim; i++)
    {
        rctd_mul(&tmp[thread_index], get_ctdvector_i(a, i), get_ctdvector_i(b, i));
    #pragma omp critical 
        rctd_add(ret, ret, &tmp[thread_index]);
    }
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_ctdmatrix(CTDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
    _bncomp_row_swap_tdmatrix(mat->re, row_index0, row_index1, col_start, col_end);
    _bncomp_row_swap_tdmatrix(mat->im, row_index0, row_index1, col_start, col_end);
}

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b)
{
    _bncomp_add_tdmatrix(c->re, a->re, b->re);
    _bncomp_add_tdmatrix(c->im, a->im, b->im);
}

/* c := a - b */
void _bncomp_sub_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b)
{
    _bncomp_sub_tdmatrix(c->re, a->re, b->re);
    _bncomp_sub_tdmatrix(c->im, a->im, b->im);
}

/* c := sc * a */
void _bncomp_cmul_ctdmatrix(CTDMatrix c, ctdfloat *sc, CTDMatrix a)
{
    TDMatrix t1, t2, t3;
    tdfloat tmp;

    t1 = init_tdmatrix(c->re->row_dim, c->re->col_dim);
    t2 = init_tdmatrix(c->re->row_dim, c->re->col_dim);

    _bncomp_cmul_tdmatrix(t1, sc->val_re, a->re);
    _bncomp_cmul_tdmatrix(t2, sc->val_im, a->im);
    _bncomp_sub_tdmatrix(c->re, t1, t2);

    #ifdef USE_4M
        // 4M
        _bncomp_cmul_tdmatrix(t1, sc->val_im, a->re);
        _bncomp_cmul_tdmatrix(t2, sc->val_re, a->im);
        _bncomp_add_tdmatrix(c->im, t1, t2);
    #else // USE_4M
        // 3M
        rtd_add(tmp.val, sc->val_re, sc->val_im);
        t3 = init_tdmatrix(c->re->row_dim, c->im->col_dim);
        _bncomp_add_tdmatrix(t3, a->re, a->im);
        _bncomp_cmul_tdmatrix(c->im, tmp.val, t3);
        _bncomp_sub_tdmatrix(c->im, c->im, t1);
        _bncomp_sub_tdmatrix(c->im, c->im, t2);
        free_tdmatrix(t3);
    #endif // USE_4M
}

/* c := a */
void _bncomp_subst_ctdmatrix(CTDMatrix c, CTDMatrix a)
{
    _bncomp_subst_tdmatrix(c->re, a->re);
    _bncomp_subst_tdmatrix(c->im, a->im);
}

/* c := I */
void _bncomp_setI_ctdmatrix(CTDMatrix c)
{
    _bncomp_setI_tdmatrix(c->re);
    _bncomp_set0_tdmatrix(c->im);
}

// set a zero matrix
//void set0_tdmatrix(TDMatrix mat)
void _bncomp_set0_ctdmatrix(CTDMatrix mat)
{
    _bncomp_set0_tdmatrix(mat->re);
    _bncomp_set0_tdmatrix(mat->im);
}

/* v := a * vb */
void _bncomp_mul_ctdmatrix_ctdvec_4m(CTDVector v, CTDMatrix a, CTDVector vb)
{
    TDVector t1, t2, t3, t4;
    TDMatrix tmp_mat;

    t1 = init_tdvector(v->re->dim);
    t2 = init_tdvector(v->re->dim);
    t3 = init_tdvector(v->re->dim);
    t4 = init_tdvector(v->re->dim);

    _bncomp_mul_tdmatrix_tdvec(t1, a->re, vb->re);
    _bncomp_mul_tdmatrix_tdvec(t2, a->im, vb->im);
    _bncomp_sub_tdvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_mul_tdmatrix_tdvec(t3, a->im, vb->re);
        _bncomp_mul_tdmatrix_tdvec(t4, a->re, vb->im);
        _bncomp_add_tdvector(v->im, t3, t4);
    //#else // USE_4M
    /*
        // 3M
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
void _bncomp_mul_ctdmatrix_ctdvec_3m(CTDVector v, CTDMatrix a, CTDVector vb)
{
    TDVector t1, t2, t3, t4;
    TDMatrix tmp_mat;

    t1 = init_tdvector(v->re->dim);
    t2 = init_tdvector(v->re->dim);
    t3 = init_tdvector(v->re->dim);
    t4 = init_tdvector(v->re->dim);

    _bncomp_mul_tdmatrix_tdvec(t1, a->re, vb->re);
    _bncomp_mul_tdmatrix_tdvec(t2, a->im, vb->im);
    _bncomp_sub_tdvector(v->re, t1, t2);

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
        _bncomp_add_tdmatrix(tmp_mat, a->re, a->im);
        _bncomp_add_tdvector(t3, vb->re, vb->im);
        _bncomp_mul_tdmatrix_tdvec(t4, tmp_mat, t3);
        _bncomp_sub_tdvector(v->im, t4, t1);
        _bncomp_sub_tdvector(v->im, v->im, t2);
        free_tdmatrix(tmp_mat);
    //#endif // USE_4M

    free_tdvector(t1);
    free_tdvector(t2);
    free_tdvector(t3);
    free_tdvector(t4);
}

/* v := a^T * vb */
void _bncomp_mul_ctdmatrixt_ctdvec(CTDVector v, CTDMatrix a, CTDVector vb)
{
    TDVector t1, t2, t3, t4;
    TDMatrix tmp_mat;

    t1 = init_tdvector(v->re->dim);
    t2 = init_tdvector(v->re->dim);
    t3 = init_tdvector(v->re->dim);
    t4 = init_tdvector(v->re->dim);

    _bncomp_mul_tdmatrixt_tdvec(t1, a->re, vb->re);
    _bncomp_mul_tdmatrixt_tdvec(t2, a->im, vb->im);
    _bncomp_sub_tdvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_mul_tdmatrixt_tdvec(t3, a->im, vb->re);
        _bncomp_mul_tdmatrixt_tdvec(t4, a->re, vb->im);
        _bncomp_add_tdvector(v->im, t3, t4);
    /*
    #else // USE_4M
        // 3M
        tmp_mat = init_tdmatrix(a->re->row_dim, a->re->col_dim);
        _bncomp_add_tdmatrix(tmp_mat, a->re, a->im);
        _bncomp_add_tdvector(t3, vb->re, vb->im);
        _bncomp_mul_tdmatrixt_tdvec(t4, tmp_mat, t3);
        _bncomp_sub_tdvector(v->im, t4, t1);
        _bncomp_sub_tdvector(v->im, v->im, t2);
        free_tdmatrix(tmp_mat);
    #endif // USE_4M
    */

    free_tdvector(t1);
    free_tdvector(t2);
    free_tdvector(t3);
    free_tdvector(t4);
}

/* v := conj(a)^T * vb */
void _bncomp_mul_ctdmatrixs_ctdvec(CTDVector v, CTDMatrix a, CTDVector vb)
{
    TDVector t1, t2, t3, t4;
    TDMatrix tmp_mat;

    t1 = init_tdvector(v->re->dim);
    t2 = init_tdvector(v->re->dim);
    t3 = init_tdvector(v->re->dim);
    t4 = init_tdvector(v->re->dim);

    _bncomp_mul_tdmatrixt_tdvec(t1, a->re, vb->re);
    _bncomp_mul_tdmatrixt_tdvec(t2, a->im, vb->im);
    _bncomp_add_tdvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_mul_tdmatrixt_tdvec(t3, a->im, vb->re);
        _bncomp_mul_tdmatrixt_tdvec(t4, a->re, vb->im);
        _bncomp_sub_tdvector(v->im, t3, t4);
    /*
    #else // USE_4M
        // 3M
        tmp_mat = init_tdmatrix(a->re->row_dim, a->re->col_dim);
        _bncomp_add_tdmatrix(tmp_mat, a->re, a->im);
        _bncomp_add_tdvector(t3, vb->re, vb->im);
        _bncomp_mul_tdmatrixt_tdvec(t4, tmp_mat, t3);
        _bncomp_sub_tdvector(v->im, t4, t1);
        _bncomp_sub_tdvector(v->im, v->im, t2);
        free_tdmatrix(tmp_mat);
    #endif // USE_4M
    */

    free_tdvector(t1);
    free_tdvector(t2);
    free_tdvector(t3);
    free_tdvector(t4);
}


// Fit dimension to be multiple of min_dim
void _bncomp_mul_ctdmatrix_oz_3m(CTDMatrix ret, CTDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CTDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    TDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    if(\
        (ret->re->row_dim != a->re->row_dim) || \
        (ret->re->col_dim != b->re->col_dim) || \
        (a->re->col_dim != b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_ctdmatrix_oz_3m\n");
        return;
    }
    t1 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tdmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_tdmatrix(b->re->row_dim, ret->re->col_dim);

    _bncomp_mul_tdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_tdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_tdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_tdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_tdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        add_tdmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_tdmatrix(t3, a->re, a->im);
        _bncomp_add_tdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        _bncomp_mul_tdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        _bncomp_sub_tdmatrix(ret->im, ret->im, t1);
        _bncomp_sub_tdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_tdmatrix(t1);
    free_tdmatrix(t2);
    free_tdmatrix(t3);
    free_tdmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void _bncomp_mul_ctdmatrix_oz_4m(CTDMatrix ret, CTDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CTDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    TDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    if(\
        (ret->re->row_dim != a->re->row_dim) || \
        (ret->re->col_dim != b->re->col_dim) || \
        (a->re->col_dim != b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_ctdmatrix_oz_4m\n");
        return;
    }
    t1 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_tdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_tdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_tdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        _bncomp_mul_tdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        _bncomp_mul_tdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        _bncomp_add_tdmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_tdmatrix(t3, a->re, a->im);
        add_tdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_tdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_tdmatrix(ret->im, ret->im, t1);
        sub_tdmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_tdmatrix(t1);
    free_tdmatrix(t2);
    free_tdmatrix(t3);
    free_tdmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (3M)
void _bncomp_mul_ctdmatrix_3m(CTDMatrix ret, CTDMatrix a, CTDMatrix b)
{
    TDMatrix t1, t2, t3, t4;
 
     if(\
        (ret->re->row_dim != a->re->row_dim) || \
        (ret->re->col_dim != b->re->col_dim) || \
        (a->re->col_dim != b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_ctdmatrix_3m\n");
        return;
    }
    t1 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tdmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_tdmatrix(b->re->row_dim, ret->re->col_dim);

    _bncomp_mul_tdmatrix(t1, a->re, b->re);
    _bncomp_mul_tdmatrix(t2, a->im, b->im);
    _bncomp_sub_tdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        _bncomp_mul_tdmatrix(t3, a->im, b->re);
        _bncomp_mul_tdmatrix(t4, a->re, b->im);
        _bncomp_add_tdmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_tdmatrix(t3, a->re, a->im);
        _bncomp_add_tdmatrix(t4, b->re, b->im);
        _bncomp_mul_tdmatrix(ret->im, t3, t4);
        _bncomp_sub_tdmatrix(ret->im, ret->im, t1);
        _bncomp_sub_tdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_tdmatrix(t1);
    free_tdmatrix(t2);
    free_tdmatrix(t3);
    free_tdmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (4M)
void _bncomp_mul_ctdmatrix_4m(CTDMatrix ret, CTDMatrix a, CTDMatrix b)
{
    TDMatrix t1, t2, t3, t4;
 
     if(\
        (ret->re->row_dim != a->re->row_dim) || \
        (ret->re->col_dim != b->re->col_dim) || \
        (a->re->col_dim != b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_ctdmatrix_4m\n");
        return;
    }
    t1 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_tdmatrix(t1, a->re, b->re);
    _bncomp_mul_tdmatrix(t2, a->im, b->im);
    _bncomp_sub_tdmatrix(ret->re, t1, t2);

    // 4M
    
    //#ifdef USE_4M
        _bncomp_mul_tdmatrix(t3, a->im, b->re);
        _bncomp_mul_tdmatrix(t4, a->re, b->im);
        _bncomp_add_tdmatrix(ret->im, t3, t4);
    //#else // USE_4M
	/*
        // 3M
        _bncomp_add_tdmatrix(t3, a->re, a->im);
        _bncomp_add_tdmatrix(t4, b->re, b->im);
        _bncomp_mul_tdmatrix(ret->im, t3, t4s);
        _bncomp_sub_tdmatrix(ret->im, ret->im, t1);
        _bncomp_sub_tdmatrix(ret->im, ret->im, t2);
	*/
    //#endif // USE_4M

    free_tdmatrix(t1);
    free_tdmatrix(t2);
    free_tdmatrix(t3);
    free_tdmatrix(t4);
}

#endif // USE_TDLINEAR
