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
//#include "ddlinear.h"
#include "cqdlinear.h"

// QD

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Complex Quad-double Precision)  */
/*                                                          */
/*                 Ver. 0.0 2023-12-12 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CQDLUdecomp(CQDMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQDMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];
    cqdfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
	{
		rcqd_abs_qd(dmaxii, get_cqdmatrix_ij(a, i, i));
		//printf("a%ld_%ld = ", i, i); rqd_out_str(dmaxii); printf("\n");
		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CQDLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			//rcqd_div(&ctmp, get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, i));
			rcqd_inv(&ctmp1, get_cqdmatrix_ij(a, i, i));
			rcqd_mul_3m(&ctmp, get_cqdmatrix_ij(a, j, i), &ctmp1);
			set_cqdmatrix_ij(a, j, i, &ctmp);
			//printf("a%ld_%ld = ", j, i); rcqd_out_str(get_cqdmatrix_ij(a, j, i)); printf("\n");
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcqd_mul_3m(&ctmp1, get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, k));
				rcqd_sub(&ctmp, get_cqdmatrix_ij(a, j, k), &ctmp1);
				set_cqdmatrix_ij(a, j, k, &ctmp);
				//printf("a%ld_%ld = ", j, k); rcqd_out_str(&ctmp); printf("\n");
			}
		}
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                (LU Decomposision of Square Dense Matrix) */
/*                         (Complex Quad-double Precision)  */
/*                                                          */
/*                 Ver. 0.0 2023-12-12 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCQDLS(CQDVector answer, CQDMatrix lu, CQDVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CQDMatrix lu: LU decomposed Matrix (given by user)  */
/*      CQDVector b: constant vector (given by user)         */
/*      CQDVector answer: Solution for linear system         */
/*      long int dim: Dimension of Matrix (given by user)   */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp[QDSIZE], dtmp1[QDSIZE];
    cqdfloat ctmp, ctmp1;

	dim = answer->re->dim;

	subst_cqdvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcqd_abs_qd(dtmp, get_cqdmatrix_ij(lu, i, i));
		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCQDLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcqd_mul_3m(&ctmp1, get_cqdmatrix_ij(lu, j, i), get_cqdvector_i(answer, i));
			rcqd_sub(&ctmp, get_cqdvector_i(answer, j), &ctmp1);
			set_cqdvector_i(answer, j, &ctmp);
		}
		//printf("f %ld = ", i); rcqd_out_str(get_cqdvector_i(answer, i)); printf("\n");
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcqd_mul_3m(&ctmp1, get_cqdmatrix_ij(lu, i, j), get_cqdvector_i(answer, j));
			rcqd_sub(&ctmp, get_cqdvector_i(answer, i), &ctmp1);
			set_cqdvector_i(answer, i, &ctmp);
		}
		//rcqd_div(&ctmp, get_cqdvector_i(answer, i), get_cqdmatrix_ij(lu, i, i));
		rcqd_inv(&ctmp1, get_cqdmatrix_ij(lu, i, i));
		rcqd_mul_3m(&ctmp, get_cqdvector_i(answer, i), &ctmp1);
		set_cqdvector_i(answer, i, &ctmp);
		//printf("b %ld = ", i); rcqd_out_str(get_cqdvector_i(answer, i)); printf("\n");
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Complex Quad-double Precision)  */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2023-12-12 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CQDLUdecompP(CQDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];
    cqdfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		rcqd_abs_qd(dmaxii, get_cqdmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rcqd_abs_qd(dtmp, get_cqdmatrix_ij(a, ch[j], i));
			if(rqd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqd_set(dmaxii, dtmp);
			}
		}

		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CQDLUdecompP!\n", i);
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
			rcqd_div(&ctmp, get_cqdmatrix_ij(a, ch[j], i), get_cqdmatrix_ij(a, ch[i], i));
			set_cqdmatrix_ij(a, ch[j], i, &ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcqd_mul(&ctmp1, get_cqdmatrix_ij(a, ch[j], i), get_cqdmatrix_ij(a, ch[i], k));
				rcqd_sub(&ctmp, get_cqdmatrix_ij(a, ch[j], k), &ctmp1);
				set_cqdmatrix_ij(a, ch[j], k, &ctmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Complex Quad-double Precision)  */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2023-12-12 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCQDLSP(CQDVector answer, CQDMatrix lu, CQDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CQDMatrix lu[]: LU decomposed Matrix (given by user)*/
/*      CQDVector b[]: constant vector (given by user)      */
/*      CQDVector answer[]: Solution for linear system      */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp[QDSIZE], dtmp1[QDSIZE];
    cqdfloat ctmp, ctmp1;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_cqdvector_i(answer, i, get_cqdvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcqd_abs_qd(dtmp, get_cqdmatrix_ij(lu, ch[i], i));
		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCQDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcqd_mul(&ctmp1, get_cqdmatrix_ij(lu, ch[j], i), get_cqdvector_i(answer, i));
			rcqd_sub(&ctmp, get_cqdvector_i(answer, j), &ctmp1);
			set_cqdvector_i(answer, j, &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcqd_mul(&ctmp1, get_cqdmatrix_ij(lu, ch[i], j), get_cqdvector_i(answer, j));
			rcqd_sub(&ctmp, get_cqdvector_i(answer, i), &ctmp1);
			set_cqdvector_i(answer, i, &ctmp);
		}
		rcqd_div(&ctmp, get_cqdvector_i(answer, i), get_cqdmatrix_ij(lu, ch[i], i));
		set_cqdvector_i(answer, i, &ctmp);
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Complex Quad-double Precision)   */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CQDLUdecompC(CQDMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQDMatrix a[]: Matrix (given by user)              */
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
	double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];
    cqdfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		rcqd_abs_qd(dmaxii, get_cqdmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcqd_abs_qd(dtmp, get_cqdmatrix_ij(a, row_ch[j], col_ch[k]));
				if(rqd_cmp(dtmp, dmaxii) > 0)
				{
					imax = j;
					jmax = k;
					rqd_set(dmaxii, dtmp);
				}
			}
		}

		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CQDLUdecompC)!\n", i);
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
			rcqd_div(&ctmp, get_cqdmatrix_ij(a, row_ch[j], col_ch[i]), get_cqdmatrix_ij(a, row_ch[i], col_ch[i]));
			set_cqdmatrix_ij(a, row_ch[j], col_ch[i], &ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcqd_mul(&ctmp1, get_cqdmatrix_ij(a, row_ch[j], col_ch[i]), get_cqdmatrix_ij(a, row_ch[i], col_ch[k]));
				rcqd_sub(&ctmp, get_cqdmatrix_ij(a, row_ch[j], col_ch[k]), &ctmp1);
				set_cqdmatrix_ij(a, row_ch[j], col_ch[k], &ctmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Complex Quad-double Precision)  */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2023-12-12 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCQDLSC(CQDVector answer, CQDMatrix lu, CQDVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQDMatrix lu: LU decomposed Matrix (given by user) */
/*       CQDVector b: constant vector (given by user)       */
/*       CQDVector answer: Solution for linear system       */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp[QDSIZE], dtmp1[QDSIZE];
    cqdfloat ctmp, ctmp1;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_cqdvector_i(answer, col_ch[i], get_cqdvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcqd_abs_qd(dtmp, get_cqdmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCQDLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcqd_mul(&ctmp1, get_cqdmatrix_ij(lu, row_ch[j], col_ch[i]), get_cqdvector_i(answer, col_ch[i]));
			rcqd_sub(&ctmp, get_cqdvector_i(answer, col_ch[j]), &ctmp1);
			set_cqdvector_i(answer, col_ch[j], &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcqd_mul(&ctmp1, get_cqdmatrix_ij(lu, row_ch[i], col_ch[j]), get_cqdvector_i(answer, col_ch[j]));
			rcqd_sub(&ctmp, get_cqdvector_i(answer, col_ch[i]), &ctmp1);
			set_cqdvector_i(answer, col_ch[i], &ctmp);
		}
		rcqd_div(&ctmp, get_cqdvector_i(answer, col_ch[i]), get_cqdmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_cqdvector_i(answer, col_ch[i], &ctmp);
	}

	return 0;
}


/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                      (Complex Quad double Precision)     */
/*              (Partial Pivoting with real swap of rows)   */
/*                                                          */
/*                 Ver. 0.0 2023-12-11 (Mon) Tomonori Kouya */
/*                 Ver. 0.1 2025-12-19 (Thu) Neon/AVX512    */
/*                                                          */
/************************************************************/
int CQDLUdecompPM(CQDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQDMatrix a: Matrix (given by user)                */
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
	double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];
    //
	cqdfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256d ctmp256_re[QDSIZE], aji256_re[QDSIZE], ajk256_re[QDSIZE], aik256_re[QDSIZE];
	__m256d ctmp256_im[QDSIZE], aji256_im[QDSIZE], ajk256_im[QDSIZE], aik256_im[QDSIZE];
#elif defined(__AVX512F__) // __AVX512F__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m512d ctmp512_re[QDSIZE], aji512_re[QDSIZE], ajk512_re[QDSIZE], aik512_re[QDSIZE];
	__m512d ctmp512_im[QDSIZE], aji512_im[QDSIZE], ajk512_im[QDSIZE], aik512_im[QDSIZE];
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	
	svfloat64_t ctmp_neon_re_0, ctmp_neon_re_1, ctmp_neon_re_2, ctmp_neon_re_3;
	svfloat64_t aji_neon_re_0, aji_neon_re_1, aji_neon_re_2, aji_neon_re_3;
	svfloat64_t ajk_neon_re_0, ajk_neon_re_1, ajk_neon_re_2, ajk_neon_re_3;
	svfloat64_t aik_neon_re_0, aik_neon_re_1, aik_neon_re_2, aik_neon_re_3;
	
	svfloat64_t ctmp_neon_im_0, ctmp_neon_im_1, ctmp_neon_im_2, ctmp_neon_im_3;
	svfloat64_t aji_neon_im_0, aji_neon_im_1, aji_neon_im_2, aji_neon_im_3;
	svfloat64_t ajk_neon_im_0, ajk_neon_im_1, ajk_neon_im_2, ajk_neon_im_3;
	svfloat64_t aik_neon_im_0, aik_neon_im_1, aik_neon_im_2, aik_neon_im_3;
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	float64x2_t ctmp_neon_re[QDSIZE], aji_neon_re[QDSIZE], ajk_neon_re[QDSIZE], aik_neon_re[QDSIZE];
	float64x2_t ctmp_neon_im[QDSIZE], aji_neon_im[QDSIZE], ajk_neon_im[QDSIZE], aik_neon_im[QDSIZE];
#else // normal
#endif // __AVX2__


#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	printf("CQDLUdecompPM AVX2 enabled!\n");
#elif defined(__AVX512F__) // __AVX512F__
	printf("CQDLUdecompPM AVX-512 enabled!\n");
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	printf("CQDLUdecompPM Arm Neon enabled!\n");
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	printf("CQDLUdecompPM Arm Neon enabled!\n");
#else
	printf("CQDLUdecompPM normal!\n");
#endif // __AVX2__

	//dim = a->col_dim;
    dim = a->re->col_dim;
	row_dim = a->re->row_dim;

	for(i = 0; i < row_dim; i++)
		ch[i] = i;

	for(i = 0; i < row_dim; i++)
	{
		// partial pivoting
		//rcqd_abs_qd(dmaxii, get_cqdmatrix_ij(a, i, i));
		subst_cqdmatrix_ij(&aii, a, i, i);
		rcqd_abs_qd(dmaxii, &aii); // get_cqdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < row_dim; j++)
		{
			subst_cqdmatrix_ij(&aji, a, j, i);
			//rcqd_abs_qd(dtmp, get_cqdmatrix_ij(a, j, i));
			rcqd_abs_qd(dtmp, &aji); // get_cqdmatrix_ij(a, j, i));
			if(rqd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqd_set(dmaxii, dtmp);
			}
		}

		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CQDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_cqdmatrix(a, i, imax, 0, a->re->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			//rcqd_div(&ctmp, get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, i));
			//rcqd_inv(&ctmp1, get_cqdmatrix_ij(a, i, i));
			subst_cqdmatrix_ij(&aii, a, i, i);
			rcqd_inv(&ctmp1, &aii); // get_cqdmatrix_ij(a, i, i));
			//rcqd_mul(&ctmp, get_cqdmatrix_ij(a, j, i), &ctmp1);
			subst_cqdmatrix_ij(&aji, a, j, i);
			rcqd_mul(&ctmp, &aji, &ctmp1);
			set_cqdmatrix_ij(a, j, i, &ctmp);
		}
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
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
				//rcqd_mul(&ctmp1, get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, k));
				//rcqd_sub(&ctmp, get_cqdmatrix_ij(a, j, k), &ctmp1);
				subst_cqdmatrix_ij(&aji, a, j, i);
				subst_cqdmatrix_ij(&aik, a, i, k);
				rcqd_mul(&ctmp1, &aji, &aik);
				subst_cqdmatrix_ij(&ajk, a, j, k);
				rcqd_sub(&ctmp, &ajk, &ctmp1);
				set_cqdmatrix_ij(a, j, k, &ctmp);
			} 
			//printf("head k_start, k = %ld, %ld, ", k_start, k);
//#if 0
			// middle : SIMD
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->re->real_col_dim + k;
				//rcdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				aik256_re[0] = _mm256_load_pd(&(a->re->element[0][index_ik]));
				aik256_re[1] = _mm256_load_pd(&(a->re->element[1][index_ik]));
				aik256_re[2] = _mm256_load_pd(&(a->re->element[2][index_ik]));
				aik256_re[3] = _mm256_load_pd(&(a->re->element[3][index_ik]));

				aik256_im[0] = _mm256_load_pd(&(a->im->element[0][index_ik]));
				aik256_im[1] = _mm256_load_pd(&(a->im->element[1][index_ik]));
				aik256_im[2] = _mm256_load_pd(&(a->im->element[2][index_ik]));
				aik256_im[3] = _mm256_load_pd(&(a->im->element[3][index_ik]));

				_bncavx2_rcqd_mul(ctmp256_re, ctmp256_im, aji256_re, aji256_im, aik256_re, aik256_im);
				//printf(" -- mul -- ");

				index_jk = j * a->re->real_col_dim + k;
				//rcdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				ajk256_re[0] = _mm256_load_pd(&(a->re->element[0][index_jk]));
				ajk256_re[1] = _mm256_load_pd(&(a->re->element[1][index_jk]));
				ajk256_re[2] = _mm256_load_pd(&(a->re->element[2][index_jk]));
				ajk256_re[3] = _mm256_load_pd(&(a->re->element[3][index_jk]));

				ajk256_im[0] = _mm256_load_pd(&(a->im->element[0][index_jk]));
				ajk256_im[1] = _mm256_load_pd(&(a->im->element[1][index_jk]));
				ajk256_im[2] = _mm256_load_pd(&(a->im->element[2][index_jk]));
				ajk256_im[3] = _mm256_load_pd(&(a->im->element[3][index_jk]));
				_bncavx2_rcqd_sub(ctmp256_re, ctmp256_im, ajk256_re, ajk256_im, ctmp256_re, ctmp256_im);
				//printf(" -- sub -- ");

				//set_cddmatrix_ij(a, j, k, dtmp);
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
#elif defined(__AVX512F__) // __AVX512F__
		// AVX-512 processes 8 doubles at a time
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
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
				subst_cqdmatrix_ij(&aji, a, j, i);
				subst_cqdmatrix_ij(&aik, a, i, k);
				rcqd_mul(&ctmp1, &aji, &aik);
				subst_cqdmatrix_ij(&ajk, a, j, k);
				rcqd_sub(&ctmp, &ajk, &ctmp1);
				set_cqdmatrix_ij(a, j, k, &ctmp);
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
				_bncavx512_rcqd_mul(ctmp512_re, ctmp512_im, aji512_re, aji512_im, aik512_re, aik512_im);

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
				_bncavx512_rcqd_sub(ctmp512_re, ctmp512_im, ajk512_re, ajk512_im, ctmp512_re, ctmp512_im);

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
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
		// Neon processes 2 doubles at a time
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->re->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->re->real_col_dim + i;
			// Broadcast a[j][i] (real and imaginary parts) to all elements in Neon registers
			// Real part
			aji_neon_re_0 = svdup_f64(a->re->element[0][index_ji]);
			aji_neon_re_1 = svdup_f64(a->re->element[1][index_ji]);
			aji_neon_re_2 = svdup_f64(a->re->element[2][index_ji]);
			aji_neon_re_3 = svdup_f64(a->re->element[3][index_ji]);

			// Imaginary part
			aji_neon_im_0 = svdup_f64(a->im->element[0][index_ji]);
			aji_neon_im_1 = svdup_f64(a->im->element[1][index_ji]);
			aji_neon_im_2 = svdup_f64(a->im->element[2][index_ji]);
			aji_neon_im_3 = svdup_f64(a->im->element[3][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				subst_cqdmatrix_ij(&aji, a, j, i);
				subst_cqdmatrix_ij(&aik, a, i, k);
				rcqd_mul(&ctmp1, &aji, &aik);
				subst_cqdmatrix_ij(&ajk, a, j, k);
				rcqd_sub(&ctmp, &ajk, &ctmp1);
				set_cqdmatrix_ij(a, j, k, &ctmp);
			}

			// middle: SIMD processing with Neon
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += (long int)svcntd())
			{
		svbool_t pg = svwhilelt_b64_s64((int64_t)k, (int64_t)(dim_end));
				index_ik = i * a->re->real_col_dim + k;
				// Load a[i][k] elements (real and imaginary parts)
				aik_neon_re_0 = svld1_f64(pg, &(a->re->element[0][index_ik]));
				aik_neon_re_1 = svld1_f64(pg, &(a->re->element[1][index_ik]));
				aik_neon_re_2 = svld1_f64(pg, &(a->re->element[2][index_ik]));
				aik_neon_re_3 = svld1_f64(pg, &(a->re->element[3][index_ik]));

				aik_neon_im_0 = svld1_f64(pg, &(a->im->element[0][index_ik]));
				aik_neon_im_1 = svld1_f64(pg, &(a->im->element[1][index_ik]));
				aik_neon_im_2 = svld1_f64(pg, &(a->im->element[2][index_ik]));
				aik_neon_im_3 = svld1_f64(pg, &(a->im->element[3][index_ik]));
				
				// Complex multiply: ctmp = a[j][i] * a[i][k]
				_bncsve2_rcqd_mul(pg, &ctmp_neon_re_0, &ctmp_neon_re_1, &ctmp_neon_re_2, &ctmp_neon_re_3, &ctmp_neon_im_0, &ctmp_neon_im_1, &ctmp_neon_im_2, &ctmp_neon_im_3, aji_neon_re_0, aji_neon_re_1, aji_neon_re_2, aji_neon_re_3, aji_neon_im_0, aji_neon_im_1, aji_neon_im_2, aji_neon_im_3, aik_neon_re_0, aik_neon_re_1, aik_neon_re_2, aik_neon_re_3, aik_neon_im_0, aik_neon_im_1, aik_neon_im_2, aik_neon_im_3);

				index_jk = j * a->re->real_col_dim + k;
				// Load a[j][k] elements (real and imaginary parts)
				ajk_neon_re_0 = svld1_f64(pg, &(a->re->element[0][index_jk]));
				ajk_neon_re_1 = svld1_f64(pg, &(a->re->element[1][index_jk]));
				ajk_neon_re_2 = svld1_f64(pg, &(a->re->element[2][index_jk]));
				ajk_neon_re_3 = svld1_f64(pg, &(a->re->element[3][index_jk]));

				ajk_neon_im_0 = svld1_f64(pg, &(a->im->element[0][index_jk]));
				ajk_neon_im_1 = svld1_f64(pg, &(a->im->element[1][index_jk]));
				ajk_neon_im_2 = svld1_f64(pg, &(a->im->element[2][index_jk]));
				ajk_neon_im_3 = svld1_f64(pg, &(a->im->element[3][index_jk]));
				
				// Complex subtract: ctmp = a[j][k] - ctmp
				_bncsve2_rcqd_sub(pg, &ctmp_neon_re_0, &ctmp_neon_re_1, &ctmp_neon_re_2, &ctmp_neon_re_3, &ctmp_neon_im_0, &ctmp_neon_im_1, &ctmp_neon_im_2, &ctmp_neon_im_3, ajk_neon_re_0, ajk_neon_re_1, ajk_neon_re_2, ajk_neon_re_3, ajk_neon_im_0, ajk_neon_im_1, ajk_neon_im_2, ajk_neon_im_3, ctmp_neon_re_0, ctmp_neon_re_1, ctmp_neon_re_2, ctmp_neon_re_3, ctmp_neon_im_0, ctmp_neon_im_1, ctmp_neon_im_2, ctmp_neon_im_3);

				// Store result back to a[j][k] (real and imaginary parts)
				svst1_f64(pg, &(a->re->element[0][index_jk]), ctmp_neon_re_0);
				svst1_f64(pg, &(a->re->element[1][index_jk]), ctmp_neon_re_1);
				svst1_f64(pg, &(a->re->element[2][index_jk]), ctmp_neon_re_2);
				svst1_f64(pg, &(a->re->element[3][index_jk]), ctmp_neon_re_3);

				svst1_f64(pg, &(a->im->element[0][index_jk]), ctmp_neon_im_0);
				svst1_f64(pg, &(a->im->element[1][index_jk]), ctmp_neon_im_1);
				svst1_f64(pg, &(a->im->element[2][index_jk]), ctmp_neon_im_2);
				svst1_f64(pg, &(a->im->element[3][index_jk]), ctmp_neon_im_3);
			}
		}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
		// Neon processes 2 doubles at a time
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
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
				subst_cqdmatrix_ij(&aji, a, j, i);
				subst_cqdmatrix_ij(&aik, a, i, k);
				rcqd_mul(&ctmp1, &aji, &aik);
				subst_cqdmatrix_ij(&ajk, a, j, k);
				rcqd_sub(&ctmp, &ajk, &ctmp1);
				set_cqdmatrix_ij(a, j, k, &ctmp);
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
				_bncneon_rcqd_mul(ctmp_neon_re, ctmp_neon_im, aji_neon_re, aji_neon_im, aik_neon_re, aik_neon_im);

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
				_bncneon_rcqd_sub(ctmp_neon_re, ctmp_neon_im, ajk_neon_re, ajk_neon_im, ctmp_neon_re, ctmp_neon_im);

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
				//rcqd_mul(&ctmp1, get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, k));
				//rcqd_sub(&ctmp, get_cqdmatrix_ij(a, j, k), &ctmp1);
				subst_cqdmatrix_ij(&aji, a, j, i);
				subst_cqdmatrix_ij(&aik, a, i, k);
				rcqd_mul(&ctmp1, &aji, &aik);
				subst_cqdmatrix_ij(&ajk, a, j, k);
				rcqd_sub(&ctmp, &ajk, &ctmp1);
				set_cqdmatrix_ij(a, j, k, &ctmp);
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
/*                  (Complex Double double Precision)       */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/*                 Ver. 0.0 2023-12-11 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCQDLSPM(CQDVector answer, CQDMatrix lu, CQDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*     CQDMatrix lu[]: LU decomposed Matrix (given by user) */
/*     CQDVector b[]: constant vector (given by user)       */
/*     CQDVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp[QDSIZE], dtmp1[QDSIZE];
    cqdfloat ctmp, ctmp1;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_cqdvector_i(answer, i, get_cqdvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcqd_abs_qd(dtmp, get_cqdmatrix_ij(lu, i, i));
		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCQDLSPM, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcqd_mul(&ctmp1, get_cqdmatrix_ij(lu, j, i), get_cqdvector_i(answer, i));
			rcqd_sub(&ctmp, get_cqdvector_i(answer, j), &ctmp1);
			set_cqdvector_i(answer, j, &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcqd_mul(&ctmp1, get_cqdmatrix_ij(lu, i, j), get_cqdvector_i(answer, j));
			rcqd_sub(&ctmp, get_cqdvector_i(answer, i), &ctmp1);
			set_cqdvector_i(answer, i, &ctmp);
		}
		//rcqd_div(&ctmp, get_cqdvector_i(answer, i), get_cqdmatrix_ij(lu, i, i));
		rcqd_inv(&ctmp1, get_cqdmatrix_ij(lu, i, i));
		rcqd_mul(&ctmp, get_cqdvector_i(answer, i), &ctmp1);
		set_cqdvector_i(answer, i, &ctmp);
	}

	return 0;
}