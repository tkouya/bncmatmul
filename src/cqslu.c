/********************************************************************************/
/* cqdlu.c:                                                                      */
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
/* stdio.h */
#ifndef _STDIO_H
#include <stdio.h>
#endif

/* math.h */
#ifndef _MATH_H
#include <math.h>
#endif

//#include "bnc.h"
//#include "dslinear.h"
#include "cqslinear.h"

// QD

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Complex Quad-float Precision)  */
/*                                                          */
/*                 Ver. 0.0 2023-12-12 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CQSLUdecomp(CQSMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQSMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	float dtmp[QSSIZE], dtmp1[QSSIZE], dmaxii[QSSIZE];
    cqsfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
	{
		rcqs_abs_qs(dmaxii, get_cqsmatrix_ij(a, i, i));
		//printf("a%ld_%ld = ", i, i); rqs_out_str_base(stdout, 10, 64, dmaxii); printf("\n");
		if(rqs_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CQSLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			//rcqs_div(&ctmp, get_cqsmatrix_ij(a, j, i), get_cqsmatrix_ij(a, i, i));
			rcqs_inv(&ctmp1, get_cqsmatrix_ij(a, i, i));
			rcqs_mul_3m(&ctmp, get_cqsmatrix_ij(a, j, i), &ctmp1);
			set_cqsmatrix_ij(a, j, i, &ctmp);
			//printf("a%ld_%ld = ", j, i); rcqs_out_str(get_cqsmatrix_ij(a, j, i)); printf("\n");
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcqs_mul_3m(&ctmp1, get_cqsmatrix_ij(a, j, i), get_cqsmatrix_ij(a, i, k));
				rcqs_sub(&ctmp, get_cqsmatrix_ij(a, j, k), &ctmp1);
				set_cqsmatrix_ij(a, j, k, &ctmp);
				//printf("a%ld_%ld = ", j, k); rcqs_out_str(&ctmp); printf("\n");
			}
		}
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                (LU Decomposision of Square Dense Matrix) */
/*                         (Complex Quad-float Precision)  */
/*                                                          */
/*                 Ver. 0.0 2023-12-12 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCQSLS(CQSVector answer, CQSMatrix lu, CQSVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CQSMatrix lu: LU decomposed Matrix (given by user)  */
/*      CQSVector b: constant vector (given by user)         */
/*      CQSVector answer: Solution for linear system         */
/*      long int dim: Dimension of Matrix (given by user)   */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	float dtmp[QSSIZE], dtmp1[QSSIZE];
    cqsfloat ctmp, ctmp1;

	dim = answer->re->dim;

	subst_cqsvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcqs_abs_qs(dtmp, get_cqsmatrix_ij(lu, i, i));
		if(rqs_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCQSLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcqs_mul_3m(&ctmp1, get_cqsmatrix_ij(lu, j, i), get_cqsvector_i(answer, i));
			rcqs_sub(&ctmp, get_cqsvector_i(answer, j), &ctmp1);
			set_cqsvector_i(answer, j, &ctmp);
		}
		//printf("f %ld = ", i); rcqs_out_str(get_cqsvector_i(answer, i)); printf("\n");
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcqs_mul_3m(&ctmp1, get_cqsmatrix_ij(lu, i, j), get_cqsvector_i(answer, j));
			rcqs_sub(&ctmp, get_cqsvector_i(answer, i), &ctmp1);
			set_cqsvector_i(answer, i, &ctmp);
		}
		//rcqs_div(&ctmp, get_cqsvector_i(answer, i), get_cqsmatrix_ij(lu, i, i));
		rcqs_inv(&ctmp1, get_cqsmatrix_ij(lu, i, i));
		rcqs_mul_3m(&ctmp, get_cqsvector_i(answer, i), &ctmp1);
		set_cqsvector_i(answer, i, &ctmp);
		//printf("b %ld = ", i); rcqs_out_str(get_cqsvector_i(answer, i)); printf("\n");
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Complex Quad-float Precision)  */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2023-12-12 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CQSLUdecompP(CQSMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQSMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	float dtmp[QSSIZE], dtmp1[QSSIZE], dmaxii[QSSIZE];
    cqsfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		rcqs_abs_qs(dmaxii, get_cqsmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rcqs_abs_qs(dtmp, get_cqsmatrix_ij(a, ch[j], i));
			if(rqs_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqs_set(dmaxii, dtmp);
			}
		}

		if(rqs_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CQSLUdecompP!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcqs_div(&ctmp, get_cqsmatrix_ij(a, ch[j], i), get_cqsmatrix_ij(a, ch[i], i));
			set_cqsmatrix_ij(a, ch[j], i, &ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcqs_mul(&ctmp1, get_cqsmatrix_ij(a, ch[j], i), get_cqsmatrix_ij(a, ch[i], k));
				rcqs_sub(&ctmp, get_cqsmatrix_ij(a, ch[j], k), &ctmp1);
				set_cqsmatrix_ij(a, ch[j], k, &ctmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Complex Quad-float Precision)  */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2023-12-12 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCQSLSP(CQSVector answer, CQSMatrix lu, CQSVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CQSMatrix lu[]: LU decomposed Matrix (given by user)*/
/*      CQSVector b[]: constant vector (given by user)      */
/*      CQSVector answer[]: Solution for linear system      */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	float dtmp[QSSIZE], dtmp1[QSSIZE];
    cqsfloat ctmp, ctmp1;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_cqsvector_i(answer, i, get_cqsvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcqs_abs_qs(dtmp, get_cqsmatrix_ij(lu, ch[i], i));
		if(rqs_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCQSLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcqs_mul(&ctmp1, get_cqsmatrix_ij(lu, ch[j], i), get_cqsvector_i(answer, i));
			rcqs_sub(&ctmp, get_cqsvector_i(answer, j), &ctmp1);
			set_cqsvector_i(answer, j, &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcqs_mul(&ctmp1, get_cqsmatrix_ij(lu, ch[i], j), get_cqsvector_i(answer, j));
			rcqs_sub(&ctmp, get_cqsvector_i(answer, i), &ctmp1);
			set_cqsvector_i(answer, i, &ctmp);
		}
		rcqs_div(&ctmp, get_cqsvector_i(answer, i), get_cqsmatrix_ij(lu, ch[i], i));
		set_cqsvector_i(answer, i, &ctmp);
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Complex Quad-float Precision)   */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CQSLUdecompC(CQSMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQSMatrix a[]: Matrix (given by user)              */
/*       long int row_ch[]: Row order                       */
/*       long int col_ch[]: Column order                    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*  row_ch[]: Row order                                     */
/*  col_ch[]: Column order                                  */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	float dtmp[QSSIZE], dtmp1[QSSIZE], dmaxii[QSSIZE];
    cqsfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		rcqs_abs_qs(dmaxii, get_cqsmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcqs_abs_qs(dtmp, get_cqsmatrix_ij(a, row_ch[j], col_ch[k]));
				if(rqs_cmp(dtmp, dmaxii) > 0)
				{
					imax = j;
					jmax = k;
					rqs_set(dmaxii, dtmp);
				}
			}
		}

		if(rqs_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CQSLUdecompC)!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = row_ch[imax];
			row_ch[imax] = row_ch[i];
			row_ch[i] = itmp;
		}
		if(jmax != i)
		{
			itmp = col_ch[jmax];
			col_ch[jmax] = col_ch[i];
			col_ch[i] = itmp;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcqs_div(&ctmp, get_cqsmatrix_ij(a, row_ch[j], col_ch[i]), get_cqsmatrix_ij(a, row_ch[i], col_ch[i]));
			set_cqsmatrix_ij(a, row_ch[j], col_ch[i], &ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcqs_mul(&ctmp1, get_cqsmatrix_ij(a, row_ch[j], col_ch[i]), get_cqsmatrix_ij(a, row_ch[i], col_ch[k]));
				rcqs_sub(&ctmp, get_cqsmatrix_ij(a, row_ch[j], col_ch[k]), &ctmp1);
				set_cqsmatrix_ij(a, row_ch[j], col_ch[k], &ctmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Complex Quad-float Precision)  */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2023-12-12 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCQSLSC(CQSVector answer, CQSMatrix lu, CQSVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQSMatrix lu: LU decomposed Matrix (given by user) */
/*       CQSVector b: constant vector (given by user)       */
/*       CQSVector answer: Solution for linear system       */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	float dtmp[QSSIZE], dtmp1[QSSIZE];
    cqsfloat ctmp, ctmp1;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_cqsvector_i(answer, col_ch[i], get_cqsvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcqs_abs_qs(dtmp, get_cqsmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rqs_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCQSLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcqs_mul(&ctmp1, get_cqsmatrix_ij(lu, row_ch[j], col_ch[i]), get_cqsvector_i(answer, col_ch[i]));
			rcqs_sub(&ctmp, get_cqsvector_i(answer, col_ch[j]), &ctmp1);
			set_cqsvector_i(answer, col_ch[j], &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcqs_mul(&ctmp1, get_cqsmatrix_ij(lu, row_ch[i], col_ch[j]), get_cqsvector_i(answer, col_ch[j]));
			rcqs_sub(&ctmp, get_cqsvector_i(answer, col_ch[i]), &ctmp1);
			set_cqsvector_i(answer, col_ch[i], &ctmp);
		}
		rcqs_div(&ctmp, get_cqsvector_i(answer, col_ch[i]), get_cqsmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_cqsvector_i(answer, col_ch[i], &ctmp);
	}

	return 0;
}


/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                      (Complex Quad float Precision)     */
/*              (Partial Pivoting with real swap of rows)   */
/*                                                          */
/*                 Ver. 0.0 2023-12-11 (Mon) Tomonori Kouya */
/*                 Ver. 0.1 2025-12-19 (Thu) Neon/AVX512    */
/*                                                          */
/************************************************************/
int CQSLUdecompPM(CQSMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQSMatrix a: Matrix (given by user)                */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim, row_dim;
	//
	float dtmp[QSSIZE], dtmp1[QSSIZE], dmaxii[QSSIZE];
    //
	cqsfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;
// SIMD : for copy & paste
#if 0 // __AVX2__ (disabled for single-complex)
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256d ctmp256_re[QSSIZE], aji256_re[QSSIZE], ajk256_re[QSSIZE], aik256_re[QSSIZE];
	__m256d ctmp256_im[QSSIZE], aji256_im[QSSIZE], ajk256_im[QSSIZE], aik256_im[QSSIZE];
#elif 0 // __AVX512F__ (disabled)
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m512d ctmp512_re[QSSIZE], aji512_re[QSSIZE], ajk512_re[QSSIZE], aik512_re[QSSIZE];
	__m512d ctmp512_im[QSSIZE], aji512_im[QSSIZE], ajk512_im[QSSIZE], aik512_im[QSSIZE];
#elif 0 // Arm Neon (disabled for single-complex)
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	float64x2_t ctmp_neon_re[QSSIZE], aji_neon_re[QSSIZE], ajk_neon_re[QSSIZE], aik_neon_re[QSSIZE];
	float64x2_t ctmp_neon_im[QSSIZE], aji_neon_im[QSSIZE], ajk_neon_im[QSSIZE], aik_neon_im[QSSIZE];
#else // normal
#endif // __AVX2__


#if 0 // __AVX2__ (disabled for single-complex)
#elif 0 // __AVX512F__ (disabled)
#elif 0 // Arm Neon (disabled for single-complex)
#else
#endif // __AVX2__

	//dim = a->col_dim;
    dim = a->re->col_dim;
	row_dim = a->re->row_dim;

	for(i = 0; i < row_dim; i++)
		ch[i] = i;

	for(i = 0; i < row_dim; i++)
	{
		// partial pivoting
		//rcqs_abs_qs(dmaxii, get_cqsmatrix_ij(a, i, i));
		subst_cqsmatrix_ij(&aii, a, i, i);
		rcqs_abs_qs(dmaxii, &aii); // get_cqsmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < row_dim; j++)
		{
			subst_cqsmatrix_ij(&aji, a, j, i);
			//rcqs_abs_qs(dtmp, get_cqsmatrix_ij(a, j, i));
			rcqs_abs_qs(dtmp, &aji); // get_cqsmatrix_ij(a, j, i));
			if(rqs_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqs_set(dmaxii, dtmp);
			}
		}

		if(rqs_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CQSLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_cqsmatrix(a, i, imax, 0, a->re->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			//rcqs_div(&ctmp, get_cqsmatrix_ij(a, j, i), get_cqsmatrix_ij(a, i, i));
			//rcqs_inv(&ctmp1, get_cqsmatrix_ij(a, i, i));
			subst_cqsmatrix_ij(&aii, a, i, i);
			rcqs_inv(&ctmp1, &aii); // get_cqsmatrix_ij(a, i, i));
			//rcqs_mul(&ctmp, get_cqsmatrix_ij(a, j, i), &ctmp1);
			subst_cqsmatrix_ij(&aji, a, j, i);
			rcqs_mul(&ctmp, &aji, &ctmp1);
			set_cqsmatrix_ij(a, j, i, &ctmp);
		}
// SIMD : for copy & paste
#if 0 // __AVX2__ (disabled for single-complex)
		dim_start = (long int)ceil((float)(i + 1) / (float)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->re->real_col_dim;

		//printf("real_row_dim, real_col_dim, dim, i, dim_start, dim_end = %ld, %ld, %ld, %ld, %ld, %ld\n", a->real_row_dim, a->real_col_dim, dim, i, dim_start, dim_end);
		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->re->real_col_dim + i;
			// Real part
			aji256_re[0] = _mm256_set_pd(
                a->re->element[0][index_ji],
                a->re->element[0][index_ji],
                a->re->element[0][index_ji],
                a->re->element[0][index_ji]
            );
			aji256_re[1] = _mm256_set_pd(
                a->re->element[1][index_ji],
                a->re->element[1][index_ji],
                a->re->element[1][index_ji],
                a->re->element[1][index_ji]
            );
			aji256_re[2] = _mm256_set_pd(
                a->re->element[2][index_ji],
                a->re->element[2][index_ji],
                a->re->element[2][index_ji],
                a->re->element[2][index_ji]
            );
			aji256_re[3] = _mm256_set_pd(
                a->re->element[3][index_ji],
                a->re->element[3][index_ji],
                a->re->element[3][index_ji],
                a->re->element[3][index_ji]
            );

			// Imaginary part
			aji256_im[0] = _mm256_set_pd(
                a->im->element[0][index_ji],
                a->im->element[0][index_ji],
                a->im->element[0][index_ji],
                a->im->element[0][index_ji]
            );
			aji256_im[1] = _mm256_set_pd(
                a->im->element[1][index_ji],
                a->im->element[1][index_ji],
                a->im->element[1][index_ji],
                a->im->element[1][index_ji]
            );
			aji256_im[2] = _mm256_set_pd(
                a->im->element[2][index_ji],
                a->im->element[2][index_ji],
                a->im->element[2][index_ji],
                a->im->element[2][index_ji]
            );
			aji256_im[3] = _mm256_set_pd(
                a->im->element[3][index_ji],
                a->im->element[3][index_ji],
                a->im->element[3][index_ji],
                a->im->element[3][index_ji]
            );

			// head
			//printf("start j, k= %ld, %ld, ", j, i + 1);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			//for(k = (i + 1); k < dim; k++)
			{
				//rcqs_mul(&ctmp1, get_cqsmatrix_ij(a, j, i), get_cqsmatrix_ij(a, i, k));
				//rcqs_sub(&ctmp, get_cqsmatrix_ij(a, j, k), &ctmp1);
				subst_cqsmatrix_ij(&aji, a, j, i);
				subst_cqsmatrix_ij(&aik, a, i, k);
				rcqs_mul(&ctmp1, &aji, &aik);
				subst_cqsmatrix_ij(&ajk, a, j, k);
				rcqs_sub(&ctmp, &ajk, &ctmp1);
				set_cqsmatrix_ij(a, j, k, &ctmp);
			} 
			//printf("head k_start, k = %ld, %ld, ", k_start, k);
//#if 0
			// middle : SIMD
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->re->real_col_dim + k;
				//rcds_mul(dtmp1, get_dsmatrix_ij(a, j, i), get_dsmatrix_ij(a, i, k));
				aik256_re[0] = _mm256_load_pd(&(a->re->element[0][index_ik]));
				aik256_re[1] = _mm256_load_pd(&(a->re->element[1][index_ik]));
				aik256_re[2] = _mm256_load_pd(&(a->re->element[2][index_ik]));
				aik256_re[3] = _mm256_load_pd(&(a->re->element[3][index_ik]));

				aik256_im[0] = _mm256_load_pd(&(a->im->element[0][index_ik]));
				aik256_im[1] = _mm256_load_pd(&(a->im->element[1][index_ik]));
				aik256_im[2] = _mm256_load_pd(&(a->im->element[2][index_ik]));
				aik256_im[3] = _mm256_load_pd(&(a->im->element[3][index_ik]));

				_bncavx2_rcqs_mul(ctmp256_re, ctmp256_im, aji256_re, aji256_im, aik256_re, aik256_im);
				//printf(" -- mul -- ");

				index_jk = j * a->re->real_col_dim + k;
				//rcds_sub(dtmp, get_dsmatrix_ij(a, j, k), dtmp1);
				ajk256_re[0] = _mm256_load_pd(&(a->re->element[0][index_jk]));
				ajk256_re[1] = _mm256_load_pd(&(a->re->element[1][index_jk]));
				ajk256_re[2] = _mm256_load_pd(&(a->re->element[2][index_jk]));
				ajk256_re[3] = _mm256_load_pd(&(a->re->element[3][index_jk]));

				ajk256_im[0] = _mm256_load_pd(&(a->im->element[0][index_jk]));
				ajk256_im[1] = _mm256_load_pd(&(a->im->element[1][index_jk]));
				ajk256_im[2] = _mm256_load_pd(&(a->im->element[2][index_jk]));
				ajk256_im[3] = _mm256_load_pd(&(a->im->element[3][index_jk]));
				_bncavx2_rcqs_sub(ctmp256_re, ctmp256_im, ajk256_re, ajk256_im, ctmp256_re, ctmp256_im);
				//printf(" -- sub -- ");

				//set_cdsmatrix_ij(a, j, k, dtmp);
				_mm256_store_pd(&(a->re->element[0][index_jk]), ctmp256_re[0]);
				_mm256_store_pd(&(a->re->element[1][index_jk]), ctmp256_re[1]);
				_mm256_store_pd(&(a->re->element[2][index_jk]), ctmp256_re[2]);
				_mm256_store_pd(&(a->re->element[3][index_jk]), ctmp256_re[3]);

				_mm256_store_pd(&(a->im->element[0][index_jk]), ctmp256_im[0]);
				_mm256_store_pd(&(a->im->element[1][index_jk]), ctmp256_im[1]);
				_mm256_store_pd(&(a->im->element[2][index_jk]), ctmp256_im[2]);
				_mm256_store_pd(&(a->im->element[3][index_jk]), ctmp256_im[3]);

			}
			//printf(", %ld middle", k);
		}
#elif 0 // __AVX512F__ (disabled)
		// AVX-512 processes 8 doubles at a time
		dim_start = (long int)ceil((float)(i + 1) / (float)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->re->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->re->real_col_dim + i;
			// Broadcast a[j][i] (real and imaginary parts) to all elements in AVX-512 registers
			// Real part
			aji512_re[0] = _mm512_set1_pd(a->re->element[0][index_ji]);
			aji512_re[1] = _mm512_set1_pd(a->re->element[1][index_ji]);
			aji512_re[2] = _mm512_set1_pd(a->re->element[2][index_ji]);
			aji512_re[3] = _mm512_set1_pd(a->re->element[3][index_ji]);

			// Imaginary part
			aji512_im[0] = _mm512_set1_pd(a->im->element[0][index_ji]);
			aji512_im[1] = _mm512_set1_pd(a->im->element[1][index_ji]);
			aji512_im[2] = _mm512_set1_pd(a->im->element[2][index_ji]);
			aji512_im[3] = _mm512_set1_pd(a->im->element[3][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				subst_cqsmatrix_ij(&aji, a, j, i);
				subst_cqsmatrix_ij(&aik, a, i, k);
				rcqs_mul(&ctmp1, &aji, &aik);
				subst_cqsmatrix_ij(&ajk, a, j, k);
				rcqs_sub(&ctmp, &ajk, &ctmp1);
				set_cqsmatrix_ij(a, j, k, &ctmp);
			}

			// middle: SIMD processing with AVX-512
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->re->real_col_dim + k;
				// Load a[i][k] elements (real and imaginary parts)
				aik512_re[0] = _mm512_load_pd(&(a->re->element[0][index_ik]));
				aik512_re[1] = _mm512_load_pd(&(a->re->element[1][index_ik]));
				aik512_re[2] = _mm512_load_pd(&(a->re->element[2][index_ik]));
				aik512_re[3] = _mm512_load_pd(&(a->re->element[3][index_ik]));

				aik512_im[0] = _mm512_load_pd(&(a->im->element[0][index_ik]));
				aik512_im[1] = _mm512_load_pd(&(a->im->element[1][index_ik]));
				aik512_im[2] = _mm512_load_pd(&(a->im->element[2][index_ik]));
				aik512_im[3] = _mm512_load_pd(&(a->im->element[3][index_ik]));
				
				// Complex multiply: ctmp = a[j][i] * a[i][k]
				_bncavx512_rcqs_mul(ctmp512_re, ctmp512_im, aji512_re, aji512_im, aik512_re, aik512_im);

				index_jk = j * a->re->real_col_dim + k;
				// Load a[j][k] elements (real and imaginary parts)
				ajk512_re[0] = _mm512_load_pd(&(a->re->element[0][index_jk]));
				ajk512_re[1] = _mm512_load_pd(&(a->re->element[1][index_jk]));
				ajk512_re[2] = _mm512_load_pd(&(a->re->element[2][index_jk]));
				ajk512_re[3] = _mm512_load_pd(&(a->re->element[3][index_jk]));

				ajk512_im[0] = _mm512_load_pd(&(a->im->element[0][index_jk]));
				ajk512_im[1] = _mm512_load_pd(&(a->im->element[1][index_jk]));
				ajk512_im[2] = _mm512_load_pd(&(a->im->element[2][index_jk]));
				ajk512_im[3] = _mm512_load_pd(&(a->im->element[3][index_jk]));
				
				// Complex subtract: ctmp = a[j][k] - ctmp
				_bncavx512_rcqs_sub(ctmp512_re, ctmp512_im, ajk512_re, ajk512_im, ctmp512_re, ctmp512_im);

				// Store result back to a[j][k] (real and imaginary parts)
				_mm512_store_pd(&(a->re->element[0][index_jk]), ctmp512_re[0]);
				_mm512_store_pd(&(a->re->element[1][index_jk]), ctmp512_re[1]);
				_mm512_store_pd(&(a->re->element[2][index_jk]), ctmp512_re[2]);
				_mm512_store_pd(&(a->re->element[3][index_jk]), ctmp512_re[3]);

				_mm512_store_pd(&(a->im->element[0][index_jk]), ctmp512_im[0]);
				_mm512_store_pd(&(a->im->element[1][index_jk]), ctmp512_im[1]);
				_mm512_store_pd(&(a->im->element[2][index_jk]), ctmp512_im[2]);
				_mm512_store_pd(&(a->im->element[3][index_jk]), ctmp512_im[3]);
			}
		}
#elif 0 // Arm Neon (disabled for single-complex)
		// Neon processes 2 doubles at a time
		dim_start = (long int)ceil((float)(i + 1) / (float)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->re->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->re->real_col_dim + i;
			// Broadcast a[j][i] (real and imaginary parts) to all elements in Neon registers
			// Real part
			aji_neon_re[0] = vdupq_n_f64(a->re->element[0][index_ji]);
			aji_neon_re[1] = vdupq_n_f64(a->re->element[1][index_ji]);
			aji_neon_re[2] = vdupq_n_f64(a->re->element[2][index_ji]);
			aji_neon_re[3] = vdupq_n_f64(a->re->element[3][index_ji]);

			// Imaginary part
			aji_neon_im[0] = vdupq_n_f64(a->im->element[0][index_ji]);
			aji_neon_im[1] = vdupq_n_f64(a->im->element[1][index_ji]);
			aji_neon_im[2] = vdupq_n_f64(a->im->element[2][index_ji]);
			aji_neon_im[3] = vdupq_n_f64(a->im->element[3][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				subst_cqsmatrix_ij(&aji, a, j, i);
				subst_cqsmatrix_ij(&aik, a, i, k);
				rcqs_mul(&ctmp1, &aji, &aik);
				subst_cqsmatrix_ij(&ajk, a, j, k);
				rcqs_sub(&ctmp, &ajk, &ctmp1);
				set_cqsmatrix_ij(a, j, k, &ctmp);
			}

			// middle: SIMD processing with Neon
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->re->real_col_dim + k;
				// Load a[i][k] elements (real and imaginary parts)
				aik_neon_re[0] = vld1q_f64(&(a->re->element[0][index_ik]));
				aik_neon_re[1] = vld1q_f64(&(a->re->element[1][index_ik]));
				aik_neon_re[2] = vld1q_f64(&(a->re->element[2][index_ik]));
				aik_neon_re[3] = vld1q_f64(&(a->re->element[3][index_ik]));

				aik_neon_im[0] = vld1q_f64(&(a->im->element[0][index_ik]));
				aik_neon_im[1] = vld1q_f64(&(a->im->element[1][index_ik]));
				aik_neon_im[2] = vld1q_f64(&(a->im->element[2][index_ik]));
				aik_neon_im[3] = vld1q_f64(&(a->im->element[3][index_ik]));
				
				// Complex multiply: ctmp = a[j][i] * a[i][k]
				_bncneon_rcqs_mul(ctmp_neon_re, ctmp_neon_im, aji_neon_re, aji_neon_im, aik_neon_re, aik_neon_im);

				index_jk = j * a->re->real_col_dim + k;
				// Load a[j][k] elements (real and imaginary parts)
				ajk_neon_re[0] = vld1q_f64(&(a->re->element[0][index_jk]));
				ajk_neon_re[1] = vld1q_f64(&(a->re->element[1][index_jk]));
				ajk_neon_re[2] = vld1q_f64(&(a->re->element[2][index_jk]));
				ajk_neon_re[3] = vld1q_f64(&(a->re->element[3][index_jk]));

				ajk_neon_im[0] = vld1q_f64(&(a->im->element[0][index_jk]));
				ajk_neon_im[1] = vld1q_f64(&(a->im->element[1][index_jk]));
				ajk_neon_im[2] = vld1q_f64(&(a->im->element[2][index_jk]));
				ajk_neon_im[3] = vld1q_f64(&(a->im->element[3][index_jk]));
				
				// Complex subtract: ctmp = a[j][k] - ctmp
				_bncneon_rcqs_sub(ctmp_neon_re, ctmp_neon_im, ajk_neon_re, ajk_neon_im, ctmp_neon_re, ctmp_neon_im);

				// Store result back to a[j][k] (real and imaginary parts)
				vst1q_f64(&(a->re->element[0][index_jk]), ctmp_neon_re[0]);
				vst1q_f64(&(a->re->element[1][index_jk]), ctmp_neon_re[1]);
				vst1q_f64(&(a->re->element[2][index_jk]), ctmp_neon_re[2]);
				vst1q_f64(&(a->re->element[3][index_jk]), ctmp_neon_re[3]);

				vst1q_f64(&(a->im->element[0][index_jk]), ctmp_neon_im[0]);
				vst1q_f64(&(a->im->element[1][index_jk]), ctmp_neon_im[1]);
				vst1q_f64(&(a->im->element[2][index_jk]), ctmp_neon_im[2]);
				vst1q_f64(&(a->im->element[3][index_jk]), ctmp_neon_im[3]);
			}
		}
#else // others
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				//rcqs_mul(&ctmp1, get_cqsmatrix_ij(a, j, i), get_cqsmatrix_ij(a, i, k));
				//rcqs_sub(&ctmp, get_cqsmatrix_ij(a, j, k), &ctmp1);
				subst_cqsmatrix_ij(&aji, a, j, i);
				subst_cqsmatrix_ij(&aik, a, i, k);
				rcqs_mul(&ctmp1, &aji, &aik);
				subst_cqsmatrix_ij(&ajk, a, j, k);
				rcqs_sub(&ctmp, &ajk, &ctmp1);
				set_cqsmatrix_ij(a, j, k, &ctmp);
			}
		}
#endif // __AVX2__
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                (LU Decomposed Square Dense Matrix)       */
/*                  (Complex Double float Precision)       */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/*                 Ver. 0.0 2023-12-11 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCQSLSPM(CQSVector answer, CQSMatrix lu, CQSVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*     CQSMatrix lu[]: LU decomposed Matrix (given by user) */
/*     CQSVector b[]: constant vector (given by user)       */
/*     CQSVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	float dtmp[QSSIZE], dtmp1[QSSIZE];
    cqsfloat ctmp, ctmp1;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_cqsvector_i(answer, i, get_cqsvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcqs_abs_qs(dtmp, get_cqsmatrix_ij(lu, i, i));
		if(rqs_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCQSLSPM, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcqs_mul(&ctmp1, get_cqsmatrix_ij(lu, j, i), get_cqsvector_i(answer, i));
			rcqs_sub(&ctmp, get_cqsvector_i(answer, j), &ctmp1);
			set_cqsvector_i(answer, j, &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcqs_mul(&ctmp1, get_cqsmatrix_ij(lu, i, j), get_cqsvector_i(answer, j));
			rcqs_sub(&ctmp, get_cqsvector_i(answer, i), &ctmp1);
			set_cqsvector_i(answer, i, &ctmp);
		}
		//rcqs_div(&ctmp, get_cqsvector_i(answer, i), get_cqsmatrix_ij(lu, i, i));
		rcqs_inv(&ctmp1, get_cqsmatrix_ij(lu, i, i));
		rcqs_mul(&ctmp, get_cqsvector_i(answer, i), &ctmp1);
		set_cqsvector_i(answer, i, &ctmp);
	}

	return 0;
}