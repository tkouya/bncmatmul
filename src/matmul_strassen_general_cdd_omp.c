/********************************************************************************/
/* matmul_strassen_general_cdd_omp.c:                                           */
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
#include <stdio.h>
#include <math.h>

//#include "bnc.h"
//#include "ddlinear.h"

#include "bncomp.h"

#ifdef USE_IMKL
	#include "mkl.h"
	#include "mkl_cblas.h" // for Intel Math Kernel Library
#endif

#include "matmul_strassen.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_add_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4], CDDMatrix mat_b, long int mat_b_index[4])
{
    _bncomp_add_ddmatrix_partial(ret->re, ret_index, mat_a->re, mat_a_index, mat_b->re, mat_b_index);
    _bncomp_add_ddmatrix_partial(ret->im, ret_index, mat_a->im, mat_a_index, mat_b->im, mat_b_index);
}

// partial sub
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_sub_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4], CDDMatrix mat_b, long int mat_b_index[4])
{
    _bncomp_sub_ddmatrix_partial(ret->re, ret_index, mat_a->re, mat_a_index, mat_b->re, mat_b_index);
    _bncomp_sub_ddmatrix_partial(ret->im, ret_index, mat_a->im, mat_a_index, mat_b->im, mat_b_index);
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4])
{
    _bncomp_subst_ddmatrix_partial(ret->re, ret_index, mat_a->re, mat_a_index);
    _bncomp_subst_ddmatrix_partial(ret->im, ret_index, mat_a->im, mat_a_index);
}

// partial set after checking
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_cddmatrix_partial_checked(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4])
{
    _bncomp_subst_ddmatrix_partial_checked(ret->re, ret_index, mat_a->re, mat_a_index);
    _bncomp_subst_ddmatrix_partial_checked(ret->im, ret_index, mat_a->im, mat_a_index);
}

// Block matrix multiplicaiton
void _bncomp_mul_cddmatrix_block_3m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim)
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
        (ret->re->row_dim != mat_a->re->row_dim) || \
        (ret->re->col_dim != mat_b->re->col_dim) || \
        (mat_a->re->col_dim != mat_b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cddmatrix_block_3m\n");
        return;
    }
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(mat_a->re->row_dim, mat_a->re->col_dim);
    t4 = init_ddmatrix(mat_b->re->row_dim, mat_b->re->col_dim);

    _bncomp_mul_ddmatrix_block(t1, mat_a->re, mat_b->re, min_dim);
    _bncomp_mul_ddmatrix_block(t2, mat_a->im, mat_b->im, min_dim);
    _bncomp_sub_ddmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_ddmatrix_block(t3, mat_a->im, mat_b->re, min_dim);
        mul_ddmatrix_block(t4, mat_a->re, mat_b->im, min_dim);
        add_ddmatrix(ret->im, t1, t2);
    #else // USE_4M 
    */
        // 3M
        _bncomp_add_ddmatrix(t3, mat_a->re, mat_a->im);
        _bncomp_add_ddmatrix(t4, mat_b->re, mat_b->im);
        _bncomp_mul_ddmatrix_block(ret->im, t3, t4, min_dim);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t1);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// Block matrix multiplicaiton
void _bncomp_mul_cddmatrix_block_4m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim)
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
        (ret->re->row_dim != mat_a->re->row_dim) || \
        (ret->re->col_dim != mat_b->re->col_dim) || \
        (mat_a->re->col_dim != mat_b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cddmatrix_block_4m\n");
        return;
    }
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_ddmatrix_block(t1, mat_a->re, mat_b->re, min_dim);
    _bncomp_mul_ddmatrix_block(t2, mat_a->im, mat_b->im, min_dim);
    _bncomp_sub_ddmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        _bncomp_mul_ddmatrix_block(t3, mat_a->im, mat_b->re, min_dim);
        _bncomp_mul_ddmatrix_block(t4, mat_a->re, mat_b->im, min_dim);
        _bncomp_add_ddmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_ddmatrix(t3, mat_a->re, mat_a->im);
        add_ddmatrix(t4, mat_b->re, mat_b->im);
        mul_ddmatrix_block(ret->im, t3, t4, min_dim);
        sub_ddmatrix(ret->im, ret->im, t1);
        sub_ddmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}


// Fit dimension to be multiple of min_dim
void _bncomp_mul_cddmatrix_strassen_3m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim)
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
        (ret->re->row_dim != mat_a->re->row_dim) || \
        (ret->re->col_dim != mat_b->re->col_dim) || \
        (mat_a->re->col_dim != mat_b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cddmatrix_strassen_3m\n");
        return;
    }
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(mat_a->re->row_dim, mat_a->re->col_dim);
    t4 = init_ddmatrix(mat_b->re->row_dim, mat_b->re->col_dim);

    _bncomp_mul_ddmatrix_strassen(t1, mat_a->re, mat_b->re, min_dim);
    _bncomp_mul_ddmatrix_strassen(t2, mat_a->im, mat_b->im, min_dim);
    _bncomp_sub_ddmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_ddmatrix_strassen(t3, mat_a->im, mat_b->re, min_dim);
        mul_ddmatrix_strassen(t4, mat_a->re, mat_b->im, min_dim);
        add_ddmatrix(ret->im, t1, t2);
    #else // USE_4M 
    */
        // 3M
        _bncomp_add_ddmatrix(t3, mat_a->re, mat_a->im);
        _bncomp_add_ddmatrix(t4, mat_b->re, mat_b->im);
        _bncomp_mul_ddmatrix_strassen(ret->im, t3, t4, min_dim);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t1);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cddmatrix_strassen_4m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim)
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
        (ret->re->row_dim != mat_a->re->row_dim) || \
        (ret->re->col_dim != mat_b->re->col_dim) || \
        (mat_a->re->col_dim != mat_b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cddmatrix_strassen_4m\n");
        return;
    }
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_ddmatrix_strassen(t1, mat_a->re, mat_b->re, min_dim);
    _bncomp_mul_ddmatrix_strassen(t2, mat_a->im, mat_b->im, min_dim);
    _bncomp_sub_ddmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        _bncomp_mul_ddmatrix_strassen(t3, mat_a->im, mat_b->re, min_dim);
        _bncomp_mul_ddmatrix_strassen(t4, mat_a->re, mat_b->im, min_dim);
        _bncomp_add_ddmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        _bncomp_add_ddmatrix(t3, mat_a->re, mat_a->im);
        _bncomp_add_ddmatrix(t4, mat_b->re, mat_b->im);
        _bncomp_mul_ddmatrix_strassen(ret->im, t3, t4, min_dim);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t1);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

#ifdef __cplusplus
} //extern "C" {
#endif // __cplusplus
