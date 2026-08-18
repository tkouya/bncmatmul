/********************************************************************************/
/* matmul_strassen_general_cqd.c:                                               */
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
void add_cqdmatrix_partial(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4], CQDMatrix mat_b, long int mat_b_index[4])
{
	long int i, j, ret_i, ret_j, a_i, a_j, b_i, b_j;
	long int imax, jmax;
	//double tmp_val[QDSIZE];
    cqdfloat tmp_val;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	//set0_qd(tmp_val);
    set0_cqd(&tmp_val);

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

			//tmp_val = get_qdmatrix_ij(mat_a, a_i, a_j) + get_qdmatrix_ij(mat_b, b_i, b_j);
			rcqd_add(&tmp_val, get_cqdmatrix_ij(mat_a, a_i, a_j), get_cqdmatrix_ij(mat_b, b_i, b_j));
			set_cqdmatrix_ij(ret, ret_i, ret_j, &tmp_val);
		}
	}
}

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_cqdmatrix_partial(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4], CQDMatrix mat_b, long int mat_b_index[4])
{
	long int i, j, ret_i, ret_j, a_i, a_j, b_i, b_j;
	long int imax, jmax;
	//double tmp_val[QDSIZE];
    cqdfloat tmp_val;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	set0_cqd(&tmp_val);

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

			//tmp_val = get_qdmatrix_ij(mat_a, a_i, a_j) - get_qdmatrix_ij(mat_b, b_i, b_j);
			rcqd_sub(&tmp_val, get_cqdmatrix_ij(mat_a, a_i, a_j), get_cqdmatrix_ij(mat_b, b_i, b_j));
			set_cqdmatrix_ij(ret, ret_i, ret_j, &tmp_val);
		}
	}
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_cqdmatrix_partial(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4])
{
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

			set_cqdmatrix_ij(ret, ret_i, ret_j, get_cqdmatrix_ij(mat_a, a_i, a_j));
		}
	}
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_cqdmatrix_partial_checked(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4])
{
	long int i, j, ret_i, ret_j, a_i, a_j;
	long int imax, jmax;
	//double *ptr_qdtmp;

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
						//ptr_qdtmp = get_qdmatrix_ij(mat_a, a_i, a_j);
						//set_qdmatrix_ij(ret, ret_i, ret_j, ptr_qdtmp);
						set_cqdmatrix_ij(ret, ret_i, ret_j, get_cqdmatrix_ij(mat_a, a_i, a_j));
					}
					else
						set0_cqdmatrix_ij(ret, ret_i, ret_j); // Padding
					//printf("Warning: ret_index = %d, %d, %d, %d\n", ret_i, ret_j, a_i, a_j);
				}
			}
		}
	}
}

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void neg_cqdmatrix_partial(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4])
{
	long int i, j, ret_i, ret_j, a_i, a_j;
	long int imax, jmax;
	//double tmp[QDSIZE];
    cqdfloat tmp;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	//SET0_QD(tmp);
    set0_cqd(&tmp);

	for(i = 0; i < imax; i++)
	{
		ret_i = ret_index[0] + i;
		a_i = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index[2] + j;
			a_j = mat_a_index[2] + j;
			rcqd_neg(&tmp, get_cqdmatrix_ij(mat_a, a_i, a_j));
			set_cqdmatrix_ij(ret, ret_i, ret_j, &tmp);
		}
	}
}

#if 0
// Simple triple-loop-way matrix multiplicaiton (3M)
void mul_cqdmatrix_3m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b)
{
    QDMatrix t1, t2, t3, t4;

    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_qdmatrix(t1, mat_a->re, mat_b->re);
    mul_qdmatrix(t2, mat_a->im, mat_b->im);
    sub_qdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_qdmatrix(t3, mat_a->im, mat_b->re);
        mul_qdmatrix(t4, mat_a->re, mat_b->im);
        add_qdmatrix(ret->im, t1, t2);
    #else // USE_4M 
    */
        // 3M
        add_qdmatrix(t3, mat_a->re, mat_a->im);
        add_qdmatrix(t4, mat_b->re, mat_b->im);
        mul_qdmatrix(ret->im, t3, t4);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// Simple triple-loop-way matrix multiplicaiton (4M)
void mul_cqdmatrix_4m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b)
{
    QDMatrix t1, t2, t3, t4;

    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_qdmatrix(t1, mat_a->re, mat_b->re);
    mul_qdmatrix(t2, mat_a->im, mat_b->im);
    sub_qdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_qdmatrix(t3, mat_a->im, mat_b->re);
        mul_qdmatrix(t4, mat_a->re, mat_b->im);
        add_qdmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_qdmatrix(t3, mat_a->re, mat_a->im);
        add_qdmatrix(t4, mat_b->re, mat_b->im);
        mul_qdmatrix(ret->im, t3, t4);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}
#endif // 0

// Block matrix multiplicaiton
void mul_cqdmatrix_block_3m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim)
{
    QDMatrix t1, t2, t3, t4;

    /*
    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, mat_a->re->col_dim);
    t4 = init_qdmatrix(mat_b->re->row_dim, ret->re->col_dim);
    */
    // ret->row_dim == mat_a->row_dim, ret->col_dim == mat_b->col_dim
    if(\
        (ret->re->row_dim != mat_a->re->row_dim) || \
        (ret->re->col_dim != mat_b->re->col_dim) || \
        (mat_a->re->col_dim != mat_b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: mul_cqdmatrix_block_3m\n");
        return;
    }
    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(mat_a->re->row_dim, mat_a->re->col_dim);
    t4 = init_qdmatrix(mat_b->re->row_dim, mat_b->re->col_dim);

    mul_qdmatrix_block(t1, mat_a->re, mat_b->re, min_dim);
    mul_qdmatrix_block(t2, mat_a->im, mat_b->im, min_dim);
    sub_qdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_qdmatrix_block(t3, mat_a->im, mat_b->re, min_dim);
        mul_qdmatrix_block(t4, mat_a->re, mat_b->im, min_dim);
        add_qdmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        add_qdmatrix(t3, mat_a->re, mat_a->im);
        add_qdmatrix(t4, mat_b->re, mat_b->im);
        mul_qdmatrix_block(ret->im, t3, t4, min_dim);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// Block matrix multiplicaiton
void mul_cqdmatrix_block_4m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim)
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
        (ret->re->row_dim != mat_a->re->row_dim) || \
        (ret->re->col_dim != mat_b->re->col_dim) || \
        (mat_a->re->col_dim != mat_b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cqdmatrix_block_3m\n");
        return;
    }
    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_qdmatrix_block(t1, mat_a->re, mat_b->re, min_dim);
    mul_qdmatrix_block(t2, mat_a->im, mat_b->im, min_dim);
    sub_qdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_qdmatrix_block(t3, mat_a->im, mat_b->re, min_dim);
        mul_qdmatrix_block(t4, mat_a->re, mat_b->im, min_dim);
        add_qdmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_qdmatrix(t3, mat_a->re, mat_a->im);
        add_qdmatrix(t4, mat_b->re, mat_b->im);
        mul_qdmatrix_block(ret->im, t3, t4, min_dim);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void mul_cqdmatrix_strassen_3m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim)
{
    QDMatrix t1, t2, t3, t4;

    /*
    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, mat_a->re->col_dim);
    t4 = init_qdmatrix(mat_b->re->row_dim, ret->re->col_dim);
    */
    // ret->row_dim == mat_a->row_dim, ret->col_dim == mat_b->col_dim
    if(\
        (ret->re->row_dim != mat_a->re->row_dim) || \
        (ret->re->col_dim != mat_b->re->col_dim) || \
        (mat_a->re->col_dim != mat_b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cqdmatrix_block_3m\n");
        return;
    }
    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(mat_a->re->row_dim, mat_a->re->col_dim);
    t4 = init_qdmatrix(mat_b->re->row_dim, mat_b->re->col_dim);

    mul_qdmatrix_strassen(t1, mat_a->re, mat_b->re, min_dim);
    mul_qdmatrix_strassen(t2, mat_a->im, mat_b->im, min_dim);
    sub_qdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_qdmatrix_strassen(t3, mat_a->im, mat_b->re, min_dim);
        mul_qdmatrix_strassen(t4, mat_a->re, mat_b->im, min_dim);
        add_qdmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        add_qdmatrix(t3, mat_a->re, mat_a->im);
        add_qdmatrix(t4, mat_b->re, mat_b->im);
        mul_qdmatrix_strassen(ret->im, t3, t4, min_dim);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void mul_cqdmatrix_strassen_4m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim)
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
        (ret->re->row_dim != mat_a->re->row_dim) || \
        (ret->re->col_dim != mat_b->re->col_dim) || \
        (mat_a->re->col_dim != mat_b->re->row_dim) \
    ) {
        fprintf(stderr, "ERROR: _bncomp_mul_cqdmatrix_block_3m\n");
        return;
    }
    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_qdmatrix_strassen(t1, mat_a->re, mat_b->re, min_dim);
    mul_qdmatrix_strassen(t2, mat_a->im, mat_b->im, min_dim);
    sub_qdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_qdmatrix_strassen(t3, mat_a->im, mat_b->re, min_dim);
        mul_qdmatrix_strassen(t4, mat_a->re, mat_b->im, min_dim);
        add_qdmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_qdmatrix(t3, mat_a->re, mat_a->im);
        add_qdmatrix(t4, mat_b->re, mat_b->im);
        mul_qdmatrix_strassen(ret->im, t3, t4, min_dim);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

#ifdef __cplusplus
} //extern "C" {
#endif // __cplusplus