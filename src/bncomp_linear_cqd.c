/********************************************************************************/
/* bncomp_linear_cqd.c: Parallelized Complex QD Precision                       */
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
// QD
//---------------------------------------

//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_cqdvector(CQDVector c, CQDVector a)
{
    _bncomp_subst_qdvector(c->re, a->re);
    _bncomp_subst_qdvector(c->im, a->im);
}

/* c = a + b */
void _bncomp_add_cqdvector(CQDVector c, CQDVector a, CQDVector b)
{
    _bncomp_add_qdvector(c->re, a->re, b->re);
    _bncomp_add_qdvector(c->im, a->im, b->im);
}

/* c = a - b */
void _bncomp_sub_cqdvector(CQDVector c, CQDVector a, CQDVector b)
{
    _bncomp_sub_qdvector(c->re, a->re, b->re);
    _bncomp_sub_qdvector(c->im, a->im, b->im);
}

/* c = val * a */
void _bncomp_cmul_cqdvector_4m(CQDVector c, cqdfloat *val, CQDVector a)
{
    QDVector t1, t2, t3;
    qdfloat tmp;

    t1 = init_qdvector(c->re->dim);
    t2 = init_qdvector(c->re->dim);

    _bncomp_cmul_qdvector(t1, val->val_re, a->re);
    _bncomp_cmul_qdvector(t2, val->val_im, a->im);
    _bncomp_sub_qdvector(c->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_cmul_qdvector(t1, val->val_im, a->re);
        _bncomp_cmul_qdvector(t2, val->val_re, a->im);
        _bncomp_add_qdvector(c->im, t1, t2);
    //#else // USE_4M
        // 3M
    /*
        rqd_add(tmp.val, val->val_re, val->val_im);
        t3 = init_qdvector(c->re->dim);
        add_qdvector(t3, a->re, a->im);
        cmul_qdvector(c->im, tmp.val, t3);
        sub_qdvector(c->im, c->im, t1);
        sub_qdvector(c->im, c->im, t2);
        free_qdvector(t3);
    */
    //#endif // USE_4M

    free_qdvector(t1);
    free_qdvector(t2);
}

/* c = val * a */
void _bncomp_cmul_cqdvector_3m(CQDVector c, cqdfloat *val, CQDVector a)
{
    QDVector t1, t2, t3;
    qdfloat tmp;

    t1 = init_qdvector(c->re->dim);
    t2 = init_qdvector(c->re->dim);

    _bncomp_cmul_qdvector(t1, val->val_re, a->re);
    _bncomp_cmul_qdvector(t2, val->val_im, a->im);
    _bncomp_sub_qdvector(c->re, t1, t2);

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
        _bncomp_add_qdvector(t3, a->re, a->im);
        _bncomp_cmul_qdvector(c->im, tmp.val, t3);
        _bncomp_sub_qdvector(c->im, c->im, t1);
        _bncomp_sub_qdvector(c->im, c->im, t2);
        free_qdvector(t3);
    //#endif // USE_4M

    free_qdvector(t1);
    free_qdvector(t2);
}

/* (a, b) */
/* (a, b) = conj(a)^T * b */
void _bncomp_ip_cqdvector(cqdfloat *ret, CQDVector a, CQDVector b)
{
    int thread_index;
    long int i;
    cqdfloat tmp[BNCOMP_MAX_NUM_THREADS], conj_a_i[BNCOMP_MAX_NUM_THREADS];

    rcqd_set0(ret);

    #pragma omp parallel for private(thread_index)
    for(i = 0; i < a->re->dim; i++)
    {
		thread_index = omp_get_thread_num();

        rcqd_conj(&conj_a_i[thread_index], get_cqdvector_i(a, i));
        rcqd_mul(&tmp[thread_index], &conj_a_i[thread_index], get_cqdvector_i(b, i));
    #pragma omp critical 
        rcqd_add(ret, ret, &tmp[thread_index]);
    }
}

/* a^T * b */
void _bncomp_dotp_cqdvector(cqdfloat *ret, CQDVector a, CQDVector b)
{
    int thread_index;
    long int i;
    cqdfloat tmp[BNCOMP_MAX_NUM_THREADS];

    rcqd_set0(ret);

    #pragma omp parallel for private(thread_index)
    for(i = 0; i < a->re->dim; i++)
    {
        rcqd_mul(&tmp[thread_index], get_cqdvector_i(a, i), get_cqdvector_i(b, i));
    #pragma omp critical 
        rcqd_add(ret, ret, &tmp[thread_index]);
    }
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_cqdmatrix(CQDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
    _bncomp_row_swap_qdmatrix(mat->re, row_index0, row_index1, col_start, col_end);
    _bncomp_row_swap_qdmatrix(mat->im, row_index0, row_index1, col_start, col_end);
}

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b)
{
    _bncomp_add_qdmatrix(c->re, a->re, b->re);
    _bncomp_add_qdmatrix(c->im, a->im, b->im);
}

/* c := a - b */
void _bncomp_sub_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b)
{
    _bncomp_sub_qdmatrix(c->re, a->re, b->re);
    _bncomp_sub_qdmatrix(c->im, a->im, b->im);
}

/* c := sc * a */
void _bncomp_cmul_cqdmatrix(CQDMatrix c, cqdfloat *sc, CQDMatrix a)
{
    QDMatrix t1, t2, t3;
    qdfloat tmp;

    t1 = init_qdmatrix(c->re->row_dim, c->re->col_dim);
    t2 = init_qdmatrix(c->re->row_dim, c->re->col_dim);

    _bncomp_cmul_qdmatrix(t1, sc->val_re, a->re);
    _bncomp_cmul_qdmatrix(t2, sc->val_im, a->im);
    _bncomp_sub_qdmatrix(c->re, t1, t2);

    #ifdef USE_4M
        // 4M
        _bncomp_cmul_qdmatrix(t1, sc->val_im, a->re);
        _bncomp_cmul_qdmatrix(t2, sc->val_re, a->im);
        _bncomp_add_qdmatrix(c->im, t1, t2);
    #else // USE_4M
        // 3M
        rqd_add(tmp.val, sc->val_re, sc->val_im);
        t3 = init_qdmatrix(c->re->row_dim, c->im->col_dim);
        _bncomp_add_qdmatrix(t3, a->re, a->im);
        _bncomp_cmul_qdmatrix(c->im, tmp.val, t3);
        _bncomp_sub_qdmatrix(c->im, c->im, t1);
        _bncomp_sub_qdmatrix(c->im, c->im, t2);
        free_qdmatrix(t3);
    #endif // USE_4M
}

/* c := a */
void _bncomp_subst_cqdmatrix(CQDMatrix c, CQDMatrix a)
{
    _bncomp_subst_qdmatrix(c->re, a->re);
    _bncomp_subst_qdmatrix(c->im, a->im);
}

/* c := I */
void _bncomp_setI_cqdmatrix(CQDMatrix c)
{
    _bncomp_setI_qdmatrix(c->re);
    _bncomp_set0_qdmatrix(c->im);
}

// set a zero matrix
//void set0_qdmatrix(QDMatrix mat)
void _bncomp_set0_cqdmatrix(CQDMatrix mat)
{
    _bncomp_set0_qdmatrix(mat->re);
    _bncomp_set0_qdmatrix(mat->im);
}

/* v := a * vb */
void _bncomp_mul_cqdmatrix_cqdvec_4m(CQDVector v, CQDMatrix a, CQDVector vb)
{
    QDVector t1, t2, t3, t4;
    QDMatrix tmp_mat;

    t1 = init_qdvector(v->re->dim);
    t2 = init_qdvector(v->re->dim);
    t3 = init_qdvector(v->re->dim);
    t4 = init_qdvector(v->re->dim);

    _bncomp_mul_qdmatrix_qdvec(t1, a->re, vb->re);
    _bncomp_mul_qdmatrix_qdvec(t2, a->im, vb->im);
    _bncomp_sub_qdvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_mul_qdmatrix_qdvec(t3, a->im, vb->re);
        _bncomp_mul_qdmatrix_qdvec(t4, a->re, vb->im);
        _bncomp_add_qdvector(v->im, t3, t4);
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
void _bncomp_mul_cqdmatrix_cqdvec_3m(CQDVector v, CQDMatrix a, CQDVector vb)
{
    QDVector t1, t2, t3, t4;
    QDMatrix tmp_mat;

    t1 = init_qdvector(v->re->dim);
    t2 = init_qdvector(v->re->dim);
    t3 = init_qdvector(v->re->dim);
    t4 = init_qdvector(v->re->dim);

    _bncomp_mul_qdmatrix_qdvec(t1, a->re, vb->re);
    _bncomp_mul_qdmatrix_qdvec(t2, a->im, vb->im);
    _bncomp_sub_qdvector(v->re, t1, t2);

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
        _bncomp_add_qdmatrix(tmp_mat, a->re, a->im);
        _bncomp_add_qdvector(t3, vb->re, vb->im);
        _bncomp_mul_qdmatrix_qdvec(t4, tmp_mat, t3);
        _bncomp_sub_qdvector(v->im, t4, t1);
        _bncomp_sub_qdvector(v->im, v->im, t2);
        free_qdmatrix(tmp_mat);
    //#endif // USE_4M

    free_qdvector(t1);
    free_qdvector(t2);
    free_qdvector(t3);
    free_qdvector(t4);
}


/* v := a^T * vb */
void _bncomp_mul_cqdmatrixt_cqdvec(CQDVector v, CQDMatrix a, CQDVector vb)
{
    QDVector t1, t2, t3, t4;
    QDMatrix tmp_mat;

    t1 = init_qdvector(v->re->dim);
    t2 = init_qdvector(v->re->dim);
    t3 = init_qdvector(v->re->dim);
    t4 = init_qdvector(v->re->dim);

    _bncomp_mul_qdmatrixt_qdvec(t1, a->re, vb->re);
    _bncomp_mul_qdmatrixt_qdvec(t2, a->im, vb->im);
    _bncomp_sub_qdvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_mul_qdmatrixt_qdvec(t3, a->im, vb->re);
        _bncomp_mul_qdmatrixt_qdvec(t4, a->re, vb->im);
        _bncomp_add_qdvector(v->im, t3, t4);
    /*
    #else // USE_4M
        // 3M
        tmp_mat = init_qdmatrix(a->re->row_dim, a->re->col_dim);
        _bncomp_add_qdmatrix(tmp_mat, a->re, a->im);
        _bncomp_add_qdvector(t3, vb->re, vb->im);
        _bncomp_mul_qdmatrixt_qdvec(t4, tmp_mat, t3);
        _bncomp_sub_qdvector(v->im, t4, t1);
        _bncomp_sub_qdvector(v->im, v->im, t2);
        free_qdmatrix(tmp_mat);
    #endif // USE_4M
    */

    free_qdvector(t1);
    free_qdvector(t2);
    free_qdvector(t3);
    free_qdvector(t4);
}

/* v := conj(a)^T * vb */
void _bncomp_mul_cqdmatrixs_cqdvec(CQDVector v, CQDMatrix a, CQDVector vb)
{
    QDVector t1, t2, t3, t4;
    QDMatrix tmp_mat;

    t1 = init_qdvector(v->re->dim);
    t2 = init_qdvector(v->re->dim);
    t3 = init_qdvector(v->re->dim);
    t4 = init_qdvector(v->re->dim);

    _bncomp_mul_qdmatrixt_qdvec(t1, a->re, vb->re);
    _bncomp_mul_qdmatrixt_qdvec(t2, a->im, vb->im);
    _bncomp_add_qdvector(v->re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_mul_qdmatrixt_qdvec(t3, a->im, vb->re);
        _bncomp_mul_qdmatrixt_qdvec(t4, a->re, vb->im);
        _bncomp_sub_qdvector(v->im, t3, t4);
    /*
    #else // USE_4M
        // 3M
        tmp_mat = init_qdmatrix(a->re->row_dim, a->re->col_dim);
        _bncomp_add_qdmatrix(tmp_mat, a->re, a->im);
        _bncomp_add_qdvector(t3, vb->re, vb->im);
        _bncomp_mul_qdmatrixt_qdvec(t4, tmp_mat, t3);
        _bncomp_sub_qdvector(v->im, t4, t1);
        _bncomp_sub_qdvector(v->im, v->im, t2);
        free_qdmatrix(tmp_mat);
    #endif // USE_4M
    */

    free_qdvector(t1);
    free_qdvector(t2);
    free_qdvector(t3);
    free_qdvector(t4);
}



// Fit dimension to be multiple of min_dim
void _bncomp_mul_cqdmatrix_oz_3m(CQDMatrix ret, CQDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    QDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    if(\
        (ret->re->row_dim != a->re->row_dim) || \
        (ret->re->col_dim != b->re->col_dim) || \
        (a->re->col_dim != b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cqdmatrix_oz_3m\n");
        return;
    }
    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(a->re->row_dim, a->re->col_dim);
    t4 = init_qdmatrix(b->re->row_dim, b->re->col_dim);

    _bncomp_mul_qdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_qdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_qdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_qdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_qdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        add_qdmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_qdmatrix(t3, a->re, a->im);
        _bncomp_add_qdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        _bncomp_mul_qdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        _bncomp_sub_qdmatrix(ret->im, ret->im, t1);
        _bncomp_sub_qdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cqdmatrix_oz_4m(CQDMatrix ret, CQDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    QDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    if(\
        (ret->re->row_dim != a->re->row_dim) || \
        (ret->re->col_dim != b->re->col_dim) || \
        (a->re->col_dim != b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cqdmatrix_oz_4m\n");
        return;
    }
    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_qdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_qdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_qdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        _bncomp_mul_qdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        _bncomp_mul_qdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        _bncomp_add_qdmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_qdmatrix(t3, a->re, a->im);
        add_qdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_qdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (3M)
void _bncomp_mul_cqdmatrix_3m(CQDMatrix ret, CQDMatrix a, CQDMatrix b)
{
    QDMatrix t1, t2, t3, t4;

    if(\
        (ret->re->row_dim != a->re->row_dim) || \
        (ret->re->col_dim != b->re->col_dim) || \
        (a->re->col_dim != b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cqdmatrix_3m\n");
        return;
    }
    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_qdmatrix(b->re->row_dim, ret->re->col_dim);

    _bncomp_mul_qdmatrix(t1, a->re, b->re);
    _bncomp_mul_qdmatrix(t2, a->im, b->im);
    _bncomp_sub_qdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        _bncomp_mul_qdmatrix(t3, a->im, b->re);
        _bncomp_mul_qdmatrix(t4, a->re, b->im);
        _bncomp_add_qdmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_qdmatrix(t3, a->re, a->im);
        _bncomp_add_qdmatrix(t4, b->re, b->im);
        _bncomp_mul_qdmatrix(ret->im, t3, t4);
        _bncomp_sub_qdmatrix(ret->im, ret->im, t1);
        _bncomp_sub_qdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (4M)
void _bncomp_mul_cqdmatrix_4m(CQDMatrix ret, CQDMatrix a, CQDMatrix b)
{
    QDMatrix t1, t2, t3, t4;
 
    /*
    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    */
    // ret->row_dim == mat_a->row_dim, ret->col_dim == mat_b->col_dim
    if(\
        (ret->re->row_dim != a->re->row_dim) || \
        (ret->re->col_dim != b->re->col_dim) || \
        (a->re->col_dim != b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cqdmatrix_4m\n");
        return;
    }
    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_qdmatrix(t1, a->re, b->re);
    _bncomp_mul_qdmatrix(t2, a->im, b->im);
    _bncomp_sub_qdmatrix(ret->re, t1, t2);

    // 4M
    
    //#ifdef USE_4M
        _bncomp_mul_qdmatrix(t3, a->im, b->re);
        _bncomp_mul_qdmatrix(t4, a->re, b->im);
        _bncomp_add_qdmatrix(ret->im, t3, t4);
    //#else // USE_4M
	/*
        // 3M
        _bncomp_add_qdmatrix(t3, a->re, a->im);
        _bncomp_add_qdmatrix(t4, b->re, b->im);
        _bncomp_mul_qdmatrix(ret->im, t3, t4s);
        _bncomp_sub_qdmatrix(ret->im, ret->im, t1);
        _bncomp_sub_qdmatrix(ret->im, ret->im, t2);
	*/
    //#endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

