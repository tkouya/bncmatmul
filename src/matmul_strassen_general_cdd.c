/********************************************************************************/
/* matmul_strassen_general_cdd.c:                                               */
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
void add_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4], CDDMatrix mat_b, long int mat_b_index[4])
{
    add_ddmatrix_partial(ret->re, ret_index, mat_a->re, mat_a_index, mat_b->re, mat_b_index);
    add_ddmatrix_partial(ret->im, ret_index, mat_a->im, mat_a_index, mat_b->im, mat_b_index);
#if 0
	long int i, j, ret_i, ret_j, a_i, a_j, b_i, b_j;
	long int imax, jmax;
	//double tmp_val[DDSIZE];
    cddfloat tmp_val;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	set0_cdd(&tmp_val); // <-- Fix! 2015-06-17 by T.Kouya

	for(i = 0; i < imax; i++)
	{
		ret_i = ret_index[0] + i;
		a_i = mat_a_index[0] + i;
		b_i = mat_b_index[0] + i;
		//printf("i: %ld %ld %ld\n", ret_i, a_i, b_i);
		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index[2] + j;
			a_j = mat_a_index[2] + j;
			b_j = mat_b_index[2] + j;
			//printf("j: %ld %ld %ld\n", ret_j, a_j, b_j);

			//tmp_val = get_ddmatrix_ij(mat_a, a_i, a_j) + get_ddmatrix_ij(mat_b, b_i, b_j);
			rcdd_add(&tmp_val, get_cddmatrix_ij(mat_a, a_i, a_j), get_cddmatrix_ij(mat_b, b_i, b_j));
			set_cddmatrix_ij(ret, ret_i, ret_j, &tmp_val);
		}
	}
#endif // 0
}

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4], CDDMatrix mat_b, long int mat_b_index[4])
{
    sub_ddmatrix_partial(ret->re, ret_index, mat_a->re, mat_a_index, mat_b->re, mat_b_index);
    sub_ddmatrix_partial(ret->im, ret_index, mat_a->im, mat_a_index, mat_b->im, mat_b_index);
#if 0
	long int i, j, ret_i, ret_j, a_i, a_j, b_i, b_j;
	long int imax, jmax;
	cddfloat tmp_val;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	set0_cdd(&tmp_val);

	for(i = 0; i < imax; i++)
	{
		ret_i = ret_index[0] + i;
		a_i = mat_a_index[0] + i;
		b_i = mat_b_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index[2] + j;
			a_j = mat_a_index[2] + j;
			b_j = mat_b_index[2] + j;

			//tmp_val = get_ddmatrix_ij(mat_a, a_i, a_j) - get_ddmatrix_ij(mat_b, b_i, b_j);
			rcdd_sub(&tmp_val, get_cddmatrix_ij(mat_a, a_i, a_j), get_cddmatrix_ij(mat_b, b_i, b_j));
			set_cddmatrix_ij(ret, ret_i, ret_j, &tmp_val);
		}
	}
#endif // 0
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4])
{
    subst_ddmatrix_partial(ret->re, ret_index, mat_a->re, mat_a_index);
    subst_ddmatrix_partial(ret->im, ret_index, mat_a->im, mat_a_index);
#if 0
	long int i, j, ret_i, ret_j, a_i, a_j;
	long int imax, jmax;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	for(i = 0; i < imax; i++)
	{
		ret_i = ret_index[0] + i;
		a_i = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index[2] + j;
			a_j = mat_a_index[2] + j;

			set_cddmatrix_ij(ret, ret_i, ret_j, get_cddmatrix_ij(mat_a, a_i, a_j));
		}
	}
#endif // 0
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_cddmatrix_partial_checked(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4])
{
    subst_ddmatrix_partial_checked(ret->re, ret_index, mat_a->re, mat_a_index);
    subst_ddmatrix_partial_checked(ret->im, ret_index, mat_a->im, mat_a_index);
#if 0
	long int i, j, ret_i, ret_j, a_i, a_j;
	long int imax, jmax;
	//double *ptr_ddtmp;
    //cddfloat cddtmp;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	for(i = 0; i < imax; i++)
	{
		ret_i = ret_index[0] + i;
		a_i = mat_a_index[0] + i;
		if((ret_i >= 0) && (ret_i < ret->re->row_dim))
		{
			for(j = 0; j < jmax; j++)
			{
				ret_j = ret_index[2] + j;
				a_j = mat_a_index[2] + j;
				if((ret_j >= 0) && (ret_j < ret->re->col_dim))
				{
					if((a_i >= 0) && (a_i < mat_a->re->row_dim) && (a_j >= 0) && (a_j < mat_a->re->col_dim))
					{
						//ptr_ddtmp = get_ddmatrix_ij(mat_a, a_i, a_j);
						//set_ddmatrix_ij(ret, ret_i, ret_j, ptr_ddtmp);
                        set_cddmatrix_ij(ret, ret_i, ret_j, get_cddmatrix_ij(mat_a, a_i, a_j));
					}
					else
						set0_cddmatrix_ij(ret, ret_i, ret_j); // Padding
					//printf("Warning: ret_index = %d, %d, %d, %d\n", ret_i, ret_j, a_i, a_j);
				}
			}
		}
	}
#endif // 0
}

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void neg_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4])
{
    neg_ddmatrix_partial(ret->re, ret_index, mat_a->re, mat_a_index);
    neg_ddmatrix_partial(ret->im, ret_index, mat_a->im, mat_a_index);
#if 0
	long int i, j, ret_i, ret_j, a_i, a_j;
	long int imax, jmax;
	//double tmp[DDSIZE];
    cddfloat ctmp;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	set0_cdd(&ctmp);

	for(i = 0; i < imax; i++)
	{
		ret_i = ret_index[0] + i;
		a_i = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index[2] + j;
			a_j = mat_a_index[2] + j;
			rcdd_neg(&ctmp, get_cddmatrix_ij(mat_a, a_i, a_j));
			set_cddmatrix_ij(ret, ret_i, ret_j, &ctmp);
		}
	}
#endif // 0
}


#if 0
// Simple triple-loop-way matrix multiplicaiton (3M)
void mul_cddmatrix_3m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b)
{
    DDMatrix t1, t2, t3, t4;

    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_ddmatrix(t1, mat_a->re, mat_b->re);
    mul_ddmatrix(t2, mat_a->im, mat_b->im);
    sub_ddmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_ddmatrix(t3, mat_a->im, mat_b->re);
        mul_ddmatrix(t4, mat_a->re, mat_b->im);
        add_ddmatrix(ret->im, t1, t2);
    #else // USE_4M 
    */
        // 3M
        add_ddmatrix(t3, mat_a->re, mat_a->im);
        add_ddmatrix(t4, mat_b->re, mat_b->im);
        mul_ddmatrix(ret->im, t3, t4);
        sub_ddmatrix(ret->im, ret->im, t1);
        sub_ddmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// Simple triple-loop-way matrix multiplicaiton (4M)
void mul_cddmatrix_4m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b)
{
    DDMatrix t1, t2, t3, t4;

    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_ddmatrix(t1, mat_a->re, mat_b->re);
    mul_ddmatrix(t2, mat_a->im, mat_b->im);
    sub_ddmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_ddmatrix(t3, mat_a->im, mat_b->re);
        mul_ddmatrix(t4, mat_a->re, mat_b->im);
        add_ddmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_ddmatrix(t3, mat_a->re, mat_a->im);
        add_ddmatrix(t4, mat_b->re, mat_b->im);
        mul_ddmatrix(ret->im, t3, t4);
        sub_ddmatrix(ret->im, ret->im, t1);
        sub_ddmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}
#endif // 0

// Block matrix multiplicaiton
void mul_cddmatrix_block_3m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim)
{
    DDMatrix t1, t2, t3, t4;

    //t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    //t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    //t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    //t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
 
    // ret->row_dim == mat_a->row_dim, ret->col_dim == mat_b->col_dim
    if(\
        (ret->re->row_dim != mat_a->re->row_dim) || \
        (ret->re->col_dim != mat_b->re->col_dim) || \
        (mat_a->re->col_dim != mat_b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: mul_cddmatrix_block_3m\n");
        return;
    }
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(mat_a->re->row_dim, mat_a->re->col_dim);
    t4 = init_ddmatrix(mat_b->re->row_dim, mat_b->re->col_dim);

    mul_ddmatrix_block(t1, mat_a->re, mat_b->re, min_dim);
    mul_ddmatrix_block(t2, mat_a->im, mat_b->im, min_dim);
    sub_ddmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_ddmatrix_block(t3, mat_a->im, mat_b->re, min_dim);
        mul_ddmatrix_block(t4, mat_a->re, mat_b->im, min_dim);
        add_ddmatrix(ret->im, t1, t2);
    #else // USE_4M 
    */
        // 3M
        //printf("block_3m: t3(%ld, %ld), t4(%ld, %ld)\n", t3->row_dim, t3->col_dim, t4->row_dim, t4->col_dim);
        add_ddmatrix(t3, mat_a->re, mat_a->im);
        add_ddmatrix(t4, mat_b->re, mat_b->im);
        mul_ddmatrix_block(ret->im, t3, t4, min_dim);
        sub_ddmatrix(ret->im, ret->im, t1);
        sub_ddmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// Block matrix multiplicaiton
void mul_cddmatrix_block_4m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim)
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
        fprintf(stderr, "ERROR: mul_cddmatrix_block_4m\n");
        return;
    }
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_ddmatrix_block(t1, mat_a->re, mat_b->re, min_dim);
    mul_ddmatrix_block(t2, mat_a->im, mat_b->im, min_dim);
    sub_ddmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_ddmatrix_block(t3, mat_a->im, mat_b->re, min_dim);
        mul_ddmatrix_block(t4, mat_a->re, mat_b->im, min_dim);
        add_ddmatrix(ret->im, t3, t4);
        //printf("block_4m: t3(%ld, %ld), t4(%ld, %ld)\n", t3->row_dim, t3->col_dim, t4->row_dim, t4->col_dim);
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
void mul_cddmatrix_strassen_3m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim)
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
        fprintf(stderr, "ERROR: mul_cddmatrix_strassen_3m\n");
        return;
    }
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(mat_a->re->row_dim, mat_a->re->col_dim);
    t4 = init_ddmatrix(mat_b->re->row_dim, mat_b->re->col_dim);

    mul_ddmatrix_strassen(t1, mat_a->re, mat_b->re, min_dim);
    mul_ddmatrix_strassen(t2, mat_a->im, mat_b->im, min_dim);
    sub_ddmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_ddmatrix_strassen(t3, mat_a->im, mat_b->re, min_dim);
        mul_ddmatrix_strassen(t4, mat_a->re, mat_b->im, min_dim);
        add_ddmatrix(ret->im, t1, t2);
    #else // USE_4M 
    */
        // 3M
        add_ddmatrix(t3, mat_a->re, mat_a->im);
        add_ddmatrix(t4, mat_b->re, mat_b->im);
        mul_ddmatrix_strassen(ret->im, t3, t4, min_dim);
        sub_ddmatrix(ret->im, ret->im, t1);
        sub_ddmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void mul_cddmatrix_strassen_4m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim)
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
        fprintf(stderr, "ERROR: mul_cddmatrix_strassen_4m\n");
        return;
    }
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_ddmatrix_strassen(t1, mat_a->re, mat_b->re, min_dim);
    mul_ddmatrix_strassen(t2, mat_a->im, mat_b->im, min_dim);
    sub_ddmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_ddmatrix_strassen(t3, mat_a->im, mat_b->re, min_dim);
        mul_ddmatrix_strassen(t4, mat_a->re, mat_b->im, min_dim);
        add_ddmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_ddmatrix(t3, mat_a->re, mat_a->im);
        add_ddmatrix(t4, mat_b->re, mat_b->im);
        mul_ddmatrix_strassen(ret->im, t3, t4, min_dim);
        sub_ddmatrix(ret->im, ret->im, t1);
        sub_ddmatrix(ret->im, ret->im, t2);
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
