/********************************************************************************/
/* cddlu.c:                                                                      */
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
#include "cddlinear.h"

// DD

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                 (Complex Double-Double Precision)        */
/*                                                          */
/*                 Ver. 0.0 2023-12-11 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CDDLUdecomp(CDDMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CDDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	//static 
	cddfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;
	//static 
	double dmaxii[DDSIZE];

	//dim = a->col_dim;
	dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
	{
		//rcdd_abs_dd(dmaxii, get_cddmatrix_ij(a, i, i));
		subst_cddmatrix_ij(&aii, a, i, i);
		rcdd_abs_dd(dmaxii, &aii); // get_cddmatrix_ij(a, i, i));
		//printf("a%ld_%ld = ", i, i); rdd_out_str(dmaxii); printf("\n");
		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CDDLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			//rcdd_nrm2_dd(dmaxii, get_cddmatrix_ij(a, i, i)); rdd_out_str(dmaxii); printf("\n");
			//rcdd_conj(&ctmp, get_cddmatrix_ij(a, j, i)); rcdd_out_str(&ctmp); printf("\n");
			subst_cddmatrix_ij(&aii, a, i, i); //rcdd_out_str(&ctmp1); printf("\n");	
			rcdd_inv(&ctmp1, &aii); // get_cddmatrix_ij(a, i, i)); //rcdd_out_str(&ctmp1); printf("\n");
			subst_cddmatrix_ij(&aji, a, j, i);
			rcdd_mul_3m(&ctmp, &aji, &ctmp1); //rcdd_out_str(&ctmp); printf("\n");
			//rcdd_div_3m(&ctmp, get_cddmatrix_ij(a, j, i), get_cddmatrix_ij(a, i, i));
			//rcdd_out_str(&ctmp); printf("\n");
			set_cddmatrix_ij(a, j, i, &ctmp);
			//printf("a%ld_%ld = ", j, i); rcdd_out_str(&ctmp); printf("\n");
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				//rcdd_mul(&ctmp1, get_cddmatrix_ij(a, j, i), get_cddmatrix_ij(a, i, k));
				//rcdd_mul_3m(&ctmp1, get_cddmatrix_ij(a, j, i), get_cddmatrix_ij(a, i, k));
				subst_cddmatrix_ij(&aji, a, j, i);
				subst_cddmatrix_ij(&aik, a, i, k);
				rcdd_mul_3m(&ctmp1, &aji, &aik); // get_cddmatrix_ij(a, j, i), get_cddmatrix_ij(a, i, k));
				//printf("a%ld_%ld= ", j, k); rcdd_out_str(&ctmp1); printf("\n");
				subst_cddmatrix_ij(&ajk, a, j, k);
				//rcdd_sub(&ctmp, get_cddmatrix_ij(a, j, k), &ctmp1);
				rcdd_sub(&ctmp, &ajk, &ctmp1);
				set_cddmatrix_ij(a, j, k, &ctmp);
				//printf("a%ld_%ld= ", j, k); rcdd_out_str(&ctmp); printf("\n");
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                      (Complex Double-double Precision)   */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCDDLS(CDDVector answer, CDDMatrix lu, CDDVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CDDMatrix lu: LU decomposed Matrix (given by user)  */
/*      CDDVector b: constant vector (given by user)        */
/*      CDDVector answer: Solution for linear system        */
/*      long int dim: Dimension of Matrix (given by user)   */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	//static 
	cddfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;
    //static 
	double dtmp[DDSIZE];

	dim = answer->re->dim;

	subst_cddvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		//printf("f %ld = ", i); rcdd_out_str(get_cddvector_i(answer, i)); printf("\n");
		rcdd_abs_dd(dtmp, get_cddmatrix_ij(lu, i, i));
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCDDLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			//rcdd_mul(&ctmp1, get_cddmatrix_ij(lu, j, i), get_cddvector_i(answer, i));
			rcdd_mul_3m(&ctmp1, get_cddmatrix_ij(lu, j, i), get_cddvector_i(answer, i));
			rcdd_sub(&ctmp, get_cddvector_i(answer, j), &ctmp1);
			set_cddvector_i(answer, j, &ctmp);
		}
		//printf("f %ld = ", i); rcdd_out_str(get_cddvector_i(answer, i)); printf("\n");
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			//rcdd_mul(&ctmp1, get_cddmatrix_ij(lu, i, j), get_cddvector_i(answer, j));
			rcdd_mul_3m(&ctmp1, get_cddmatrix_ij(lu, i, j), get_cddvector_i(answer, j));
			rcdd_sub(&ctmp, get_cddvector_i(answer, i), &ctmp1);
			set_cddvector_i(answer, i, &ctmp);
		}
		//rcdd_div(&ctmp, get_cddvector_i(answer, i), get_cddmatrix_ij(lu, i, i));
		rcdd_inv(&ctmp1, get_cddmatrix_ij(lu, i, i));
		//rcdd_mul(&ctmp, get_cddvector_i(answer, i), &ctmp1);
		rcdd_mul_3m(&ctmp, get_cddvector_i(answer, i), &ctmp1);
		set_cddvector_i(answer, i, &ctmp);
		//printf("b %ld = ", i); rcdd_out_str(get_cddvector_i(answer, i)); printf("\n");
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                               (Double-Double Precision)  */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CDDLUdecompP(CDDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	static cddfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;
    static double dmaxii[DDSIZE], dtmp[DDSIZE];

	//dim = a->col_dim;
    dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		rcdd_abs_dd(dmaxii, get_cddmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rcdd_abs_dd(dtmp, get_cddmatrix_ij(a, ch[j], i));
			if(rdd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rdd_set(dmaxii, dtmp);
			}
		}

		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CDDLUdecompP!\n", i);
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
			//rcdd_div(&ctmp, get_cddmatrix_ij(a, ch[j], i), get_cddmatrix_ij(a, ch[i], i));
			rcdd_inv(&ctmp1, get_cddmatrix_ij(a, ch[i], i));
			rcdd_mul(&ctmp, get_cddmatrix_ij(a, ch[j], i), &ctmp1);
			set_cddmatrix_ij(a, ch[j], i, &ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcdd_mul(&ctmp1, get_cddmatrix_ij(a, ch[j], i), get_cddmatrix_ij(a, ch[i], k));
				rcdd_sub(&ctmp, get_cddmatrix_ij(a, ch[j], k), &ctmp1);
				set_cddmatrix_ij(a, ch[j], k, &ctmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                       (Complex Double-Double Precision)  */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2023-12-12 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCDDLSP(CDDVector answer, CDDMatrix lu, CDDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*     CDDMatrix lu[]: LU decomposed Matrix (given by user) */
/*     CDDVector b[]: constant vector (given by user)       */
/*     CDDVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[DDSIZE], dtmp1[DDSIZE];
    static cddfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_cddvector_i(answer, i, get_cddvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcdd_abs_dd(dtmp, get_cddmatrix_ij(lu, ch[i], i));
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCDDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcdd_mul(&ctmp1, get_cddmatrix_ij(lu, ch[j], i), get_cddvector_i(answer, i));
			rcdd_sub(&ctmp, get_cddvector_i(answer, j), &ctmp1);
			set_cddvector_i(answer, j, &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcdd_mul(&ctmp1, get_cddmatrix_ij(lu, ch[i], j), get_cddvector_i(answer, j));
			rcdd_sub(&ctmp, get_cddvector_i(answer, i), &ctmp1);
			set_cddvector_i(answer, i, &ctmp);
		}
		//rcdd_div(&ctmp, get_cddvector_i(answer, i), get_cddmatrix_ij(lu, ch[i], i));
		rcdd_inv(&ctmp1, get_cddmatrix_ij(lu, ch[i], i));
		rcdd_mul(&ctmp, get_cddvector_i(answer, i), &ctmp1);
		set_cddvector_i(answer, i, &ctmp);
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                       (Complex Double-Double Precision)  */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2023-12-11 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CDDLUdecompC(CDDMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDDMatrix a[]: Matrix (given by user)              */
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
	static double dtmp[DDSIZE], dtmp1[DDSIZE], dmaxii[DDSIZE];
    static cddfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	//dim = a->col_dim;
    dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		rcdd_abs_dd(dmaxii, get_cddmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcdd_abs_dd(dtmp, get_cddmatrix_ij(a, row_ch[j], col_ch[k]));
				if(rdd_cmp(dtmp, dmaxii) > 0)
				{
					imax = j;
					jmax = k;
					rdd_set(dmaxii, dtmp);
				}
			}
		}

		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CDDLUdecompC)!\n", i);
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
			//rcdd_div(&ctmp, get_cddmatrix_ij(a, row_ch[j], col_ch[i]), get_cddmatrix_ij(a, row_ch[i], col_ch[i]));
			rcdd_inv(&ctmp1, get_cddmatrix_ij(a, row_ch[i], col_ch[i]));
			rcdd_mul(&ctmp, get_cddmatrix_ij(a, row_ch[j], col_ch[i]), &ctmp1);
			set_cddmatrix_ij(a, row_ch[j], col_ch[i], &ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcdd_mul(&ctmp1, get_cddmatrix_ij(a, row_ch[j], col_ch[i]), get_cddmatrix_ij(a, row_ch[i], col_ch[k]));
				rcdd_sub(&ctmp, get_cddmatrix_ij(a, row_ch[j], col_ch[k]), &ctmp1);
				set_cddmatrix_ij(a, row_ch[j], col_ch[k], &ctmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                       (Complex Double-Double Precision)  */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2023-12-11 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCDDLSC(CDDVector answer, CDDMatrix lu, CDDVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDDMatrix lu: LU decomposed Matrix (given by user) */
/*       CDDVector b: constant vector (given by user)       */
/*       CDDVector answer: Solution for linear system       */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[DDSIZE], dtmp1[DDSIZE];
    static cddfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_cddvector_i(answer, col_ch[i], get_cddvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcdd_abs_dd(dtmp, get_cddmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCDDLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcdd_mul(&ctmp1, get_cddmatrix_ij(lu, row_ch[j], col_ch[i]), get_cddvector_i(answer, col_ch[i]));
			rcdd_sub(&ctmp, get_cddvector_i(answer, col_ch[j]), &ctmp1);
			set_cddvector_i(answer, col_ch[j], &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcdd_mul(&ctmp1, get_cddmatrix_ij(lu, row_ch[i], col_ch[j]), get_cddvector_i(answer, col_ch[j]));
			rcdd_sub(&ctmp, get_cddvector_i(answer, col_ch[i]), &ctmp1);
			set_cddvector_i(answer, col_ch[i], &ctmp);
		}
		//rcdd_div(&ctmp, get_cddvector_i(answer, col_ch[i]), get_cddmatrix_ij(lu, row_ch[i], col_ch[i]));
		rcdd_inv(&ctmp1, get_cddmatrix_ij(lu, row_ch[i], col_ch[i]));
		rcdd_mul(&ctmp, get_cddvector_i(answer, col_ch[i]), &ctmp1);
		set_cddvector_i(answer, col_ch[i], &ctmp);
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                      (Complex Double double Precision)   */
/*              (Partial Pivoting with real swap of rows)   */
/*                                                          */
/*                 Ver. 0.0 2023-12-11 (Mon) Tomonori Kouya */
/*                 Ver. 0.1 2025-12-19 (Thu) Neon/AVX512    */
/*                                                          */
/************************************************************/
int CDDLUdecompPM(CDDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDDMatrix a: Matrix (given by user)                */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim, row_dim;
	static double dtmp[DDSIZE], dtmp1[DDSIZE], dmaxii[DDSIZE];
    static cddfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256d ctmp256_re[DDSIZE], aji256_re[DDSIZE], ajk256_re[DDSIZE], aik256_re[DDSIZE];
	__m256d ctmp256_im[DDSIZE], aji256_im[DDSIZE], ajk256_im[DDSIZE], aik256_im[DDSIZE];
#elif defined(__AVX512F__) // __AVX512F__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m512d ctmp512_re[DDSIZE], aji512_re[DDSIZE], ajk512_re[DDSIZE], aik512_re[DDSIZE];
	__m512d ctmp512_im[DDSIZE], aji512_im[DDSIZE], ajk512_im[DDSIZE], aik512_im[DDSIZE];
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	
	svfloat64_t ctmp_neon_re_0, ctmp_neon_re_1;
	svfloat64_t aji_neon_re_0, aji_neon_re_1;
	svfloat64_t ajk_neon_re_0, ajk_neon_re_1;
	svfloat64_t aik_neon_re_0, aik_neon_re_1;
	
	svfloat64_t ctmp_neon_im_0, ctmp_neon_im_1;
	svfloat64_t aji_neon_im_0, aji_neon_im_1;
	svfloat64_t ajk_neon_im_0, ajk_neon_im_1;
	svfloat64_t aik_neon_im_0, aik_neon_im_1;
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	float64x2_t ctmp_neon_re[DDSIZE], aji_neon_re[DDSIZE], ajk_neon_re[DDSIZE], aik_neon_re[DDSIZE];
	float64x2_t ctmp_neon_im[DDSIZE], aji_neon_im[DDSIZE], ajk_neon_im[DDSIZE], aik_neon_im[DDSIZE];
#else // normal
#endif // __AVX2__


#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	printf("CDDLUdecompPM AVX2 enabled!\n");
#elif defined(__AVX512F__) // __AVX512F__
	printf("CDDLUdecompPM AVX-512 enabled!\n");
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	printf("CDDLUdecompPM Arm Neon enabled!\n");
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	printf("CDDLUdecompPM Arm Neon enabled!\n");
#else
	printf("CDDLUdecompPM normal!\n");
#endif // __AVX2__

	//dim = a->col_dim;
    dim = a->re->col_dim;
	row_dim = a->re->row_dim;

	for(i = 0; i < row_dim; i++)
		ch[i] = i;

	for(i = 0; i < row_dim; i++)
	{
		// partial pivoting
		//printf("i = %ld -> ", i); rcdd_out_str(get_cddmatrix_ij(a, i, i)); printf("\n");
		subst_cddmatrix_ij(&aii, a, i, i);
		//printf("i = %ld -> ", i); rcdd_out_str(&ctmp); printf("\n");
		//rcdd_abs_dd(dmaxii, get_cddmatrix_ij(a, i, i));
		rcdd_abs_dd(dmaxii, &aii);
		//printf("dmaxii = %25.17e\n", dmaxii[0]);
		imax = i;
		for(j = (i + 1); j < row_dim; j++)
		{
			//rcdd_abs_dd(dtmp, get_cddmatrix_ij(a, j, i));
			subst_cddmatrix_ij(&aji, a, j, i);
			rcdd_abs_dd(dtmp, &aji);
			if(rdd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rdd_set(dmaxii, dtmp);
			}
		}
		//printf("dmaxii = %25.17e, %ld -> %ld\n", dmaxii[0], i, imax);
		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CDDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_cddmatrix(a, i, imax, 0, a->re->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			//rcdd_div(&ctmp, get_cddmatrix_ij(a, j, i), get_cddmatrix_ij(a, i, i));
			subst_cddmatrix_ij(&aii, a, i, i);
			//rcdd_inv(&ctmp1, get_cddmatrix_ij(a, i, i));
			rcdd_inv(&ctmp1, &aii); //get_cddmatrix_ij(a, i, i));
			subst_cddmatrix_ij(&aji, a, j, i);
			//rcdd_mul(&ctmp, get_cddmatrix_ij(a, j, i), &ctmp1);
			rcdd_mul(&ctmp, &aji, &ctmp1);
			set_cddmatrix_ij(a, j, i, &ctmp);
		}
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->re->real_col_dim;

		//printf("real_row_dim, real_col_dim, dim, i, dim_start, dim_end = %ld, %ld, %ld, %ld, %ld, %ld\n", a->real_row_dim, a->real_col_dim, dim, i, dim_start, dim_end);
		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->re->real_col_dim + i;
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

			// head
			//printf("start j, k= %ld, %ld, ", j, i + 1);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			//for(k = (i + 1); k < dim; k++)
			{
				//rcdd_mul(&ctmp1, get_cddmatrix_ij(a, j, i), get_cddmatrix_ij(a, i, k));
				//rcdd_sub(&ctmp, get_cddmatrix_ij(a, j, k), &ctmp1);
				subst_cddmatrix_ij(&aji, a, j, i);
				subst_cddmatrix_ij(&aik, a, i, k);
				rcdd_mul(&ctmp1, &aji, &aik);

				subst_cddmatrix_ij(&ajk, a, j, k);
				rcdd_sub(&ctmp, &ajk, &ctmp1);
				set_cddmatrix_ij(a, j, k, &ctmp);
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
				aik256_im[0] = _mm256_load_pd(&(a->im->element[0][index_ik]));
				aik256_im[1] = _mm256_load_pd(&(a->im->element[1][index_ik]));
				_bncavx2_rcdd_mul(ctmp256_re, ctmp256_im, aji256_re, aji256_im, aik256_re, aik256_im);
				//printf(" -- mul -- ");

				index_jk = j * a->re->real_col_dim + k;
				//rcdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				ajk256_re[0] = _mm256_load_pd(&(a->re->element[0][index_jk]));
				ajk256_re[1] = _mm256_load_pd(&(a->re->element[1][index_jk]));
				ajk256_im[0] = _mm256_load_pd(&(a->im->element[0][index_jk]));
				ajk256_im[1] = _mm256_load_pd(&(a->im->element[1][index_jk]));
				_bncavx2_rcdd_sub(ctmp256_re, ctmp256_im, ajk256_re, ajk256_im, ctmp256_re, ctmp256_im);
				//printf(" -- sub -- ");

				//set_cddmatrix_ij(a, j, k, dtmp);
				_mm256_store_pd(&(a->re->element[0][index_jk]), ctmp256_re[0]);
				_mm256_store_pd(&(a->re->element[1][index_jk]), ctmp256_re[1]);
				_mm256_store_pd(&(a->im->element[0][index_jk]), ctmp256_im[0]);
				_mm256_store_pd(&(a->im->element[1][index_jk]), ctmp256_im[1]);
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
			aji512_re[0] = _mm512_set1_pd(a->re->element[0][index_ji]);
			aji512_re[1] = _mm512_set1_pd(a->re->element[1][index_ji]);
			aji512_im[0] = _mm512_set1_pd(a->im->element[0][index_ji]);
			aji512_im[1] = _mm512_set1_pd(a->im->element[1][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				subst_cddmatrix_ij(&aji, a, j, i);
				subst_cddmatrix_ij(&aik, a, i, k);
				rcdd_mul(&ctmp1, &aji, &aik);

				subst_cddmatrix_ij(&ajk, a, j, k);
				rcdd_sub(&ctmp, &ajk, &ctmp1);
				set_cddmatrix_ij(a, j, k, &ctmp);
			}

			// middle: SIMD processing with AVX-512
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->re->real_col_dim + k;
				// Load a[i][k] elements (real and imaginary parts)
				aik512_re[0] = _mm512_load_pd(&(a->re->element[0][index_ik]));
				aik512_re[1] = _mm512_load_pd(&(a->re->element[1][index_ik]));
				aik512_im[0] = _mm512_load_pd(&(a->im->element[0][index_ik]));
				aik512_im[1] = _mm512_load_pd(&(a->im->element[1][index_ik]));
				
				// Complex multiply: ctmp = a[j][i] * a[i][k]
				_bncavx512_rcdd_mul(ctmp512_re, ctmp512_im, aji512_re, aji512_im, aik512_re, aik512_im);

				index_jk = j * a->re->real_col_dim + k;
				// Load a[j][k] elements (real and imaginary parts)
				ajk512_re[0] = _mm512_load_pd(&(a->re->element[0][index_jk]));
				ajk512_re[1] = _mm512_load_pd(&(a->re->element[1][index_jk]));
				ajk512_im[0] = _mm512_load_pd(&(a->im->element[0][index_jk]));
				ajk512_im[1] = _mm512_load_pd(&(a->im->element[1][index_jk]));
				
				// Complex subtract: ctmp = a[j][k] - ctmp
				_bncavx512_rcdd_sub(ctmp512_re, ctmp512_im, ajk512_re, ajk512_im, ctmp512_re, ctmp512_im);

				// Store result back to a[j][k] (real and imaginary parts)
				_mm512_store_pd(&(a->re->element[0][index_jk]), ctmp512_re[0]);
				_mm512_store_pd(&(a->re->element[1][index_jk]), ctmp512_re[1]);
				_mm512_store_pd(&(a->im->element[0][index_jk]), ctmp512_im[0]);
				_mm512_store_pd(&(a->im->element[1][index_jk]), ctmp512_im[1]);
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
			aji_neon_re_0 = svdup_f64(a->re->element[0][index_ji]);
			aji_neon_re_1 = svdup_f64(a->re->element[1][index_ji]);
			aji_neon_im_0 = svdup_f64(a->im->element[0][index_ji]);
			aji_neon_im_1 = svdup_f64(a->im->element[1][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				subst_cddmatrix_ij(&aji, a, j, i);
				subst_cddmatrix_ij(&aik, a, i, k);
				rcdd_mul(&ctmp1, &aji, &aik);

				subst_cddmatrix_ij(&ajk, a, j, k);
				rcdd_sub(&ctmp, &ajk, &ctmp1);
				set_cddmatrix_ij(a, j, k, &ctmp);
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
				aik_neon_im_0 = svld1_f64(pg, &(a->im->element[0][index_ik]));
				aik_neon_im_1 = svld1_f64(pg, &(a->im->element[1][index_ik]));
				
				// Complex multiply: ctmp = a[j][i] * a[i][k]
				_bncsve2_rcdd_mul(pg, &ctmp_neon_re_0, &ctmp_neon_re_1, &ctmp_neon_im_0, &ctmp_neon_im_1, aji_neon_re_0, aji_neon_re_1, aji_neon_im_0, aji_neon_im_1, aik_neon_re_0, aik_neon_re_1, aik_neon_im_0, aik_neon_im_1);

				index_jk = j * a->re->real_col_dim + k;
				// Load a[j][k] elements (real and imaginary parts)
				ajk_neon_re_0 = svld1_f64(pg, &(a->re->element[0][index_jk]));
				ajk_neon_re_1 = svld1_f64(pg, &(a->re->element[1][index_jk]));
				ajk_neon_im_0 = svld1_f64(pg, &(a->im->element[0][index_jk]));
				ajk_neon_im_1 = svld1_f64(pg, &(a->im->element[1][index_jk]));
				
				// Complex subtract: ctmp = a[j][k] - ctmp
				_bncsve2_rcdd_sub(pg, &ctmp_neon_re_0, &ctmp_neon_re_1, &ctmp_neon_im_0, &ctmp_neon_im_1, ajk_neon_re_0, ajk_neon_re_1, ajk_neon_im_0, ajk_neon_im_1, ctmp_neon_re_0, ctmp_neon_re_1, ctmp_neon_im_0, ctmp_neon_im_1);

				// Store result back to a[j][k] (real and imaginary parts)
				svst1_f64(pg, &(a->re->element[0][index_jk]), ctmp_neon_re_0);
				svst1_f64(pg, &(a->re->element[1][index_jk]), ctmp_neon_re_1);
				svst1_f64(pg, &(a->im->element[0][index_jk]), ctmp_neon_im_0);
				svst1_f64(pg, &(a->im->element[1][index_jk]), ctmp_neon_im_1);
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
			aji_neon_re[0] = vdupq_n_f64(a->re->element[0][index_ji]);
			aji_neon_re[1] = vdupq_n_f64(a->re->element[1][index_ji]);
			aji_neon_im[0] = vdupq_n_f64(a->im->element[0][index_ji]);
			aji_neon_im[1] = vdupq_n_f64(a->im->element[1][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				subst_cddmatrix_ij(&aji, a, j, i);
				subst_cddmatrix_ij(&aik, a, i, k);
				rcdd_mul(&ctmp1, &aji, &aik);

				subst_cddmatrix_ij(&ajk, a, j, k);
				rcdd_sub(&ctmp, &ajk, &ctmp1);
				set_cddmatrix_ij(a, j, k, &ctmp);
			}

			// middle: SIMD processing with Neon
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->re->real_col_dim + k;
				// Load a[i][k] elements (real and imaginary parts)
				aik_neon_re[0] = vld1q_f64(&(a->re->element[0][index_ik]));
				aik_neon_re[1] = vld1q_f64(&(a->re->element[1][index_ik]));
				aik_neon_im[0] = vld1q_f64(&(a->im->element[0][index_ik]));
				aik_neon_im[1] = vld1q_f64(&(a->im->element[1][index_ik]));
				
				// Complex multiply: ctmp = a[j][i] * a[i][k]
				_bncneon_rcdd_mul(ctmp_neon_re, ctmp_neon_im, aji_neon_re, aji_neon_im, aik_neon_re, aik_neon_im);

				index_jk = j * a->re->real_col_dim + k;
				// Load a[j][k] elements (real and imaginary parts)
				ajk_neon_re[0] = vld1q_f64(&(a->re->element[0][index_jk]));
				ajk_neon_re[1] = vld1q_f64(&(a->re->element[1][index_jk]));
				ajk_neon_im[0] = vld1q_f64(&(a->im->element[0][index_jk]));
				ajk_neon_im[1] = vld1q_f64(&(a->im->element[1][index_jk]));
				
				// Complex subtract: ctmp = a[j][k] - ctmp
				_bncneon_rcdd_sub(ctmp_neon_re, ctmp_neon_im, ajk_neon_re, ajk_neon_im, ctmp_neon_re, ctmp_neon_im);

				// Store result back to a[j][k] (real and imaginary parts)
				vst1q_f64(&(a->re->element[0][index_jk]), ctmp_neon_re[0]);
				vst1q_f64(&(a->re->element[1][index_jk]), ctmp_neon_re[1]);
				vst1q_f64(&(a->im->element[0][index_jk]), ctmp_neon_im[0]);
				vst1q_f64(&(a->im->element[1][index_jk]), ctmp_neon_im[1]);
			}
		}
#else // others
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				subst_cddmatrix_ij(&ctmp, a, j, i);
				subst_cddmatrix_ij(&ctmp2, a, i, k);
				//rcdd_mul(&ctmp1, get_cddmatrix_ij(a, j, i), get_cddmatrix_ij(a, i, k));
				rcdd_mul(&ctmp1, &ctmp, &ctmp2); //get_cddmatrix_ij(a, i, k));
				//rcdd_sub(&ctmp, get_cddmatrix_ij(a, j, k), &ctmp1);
				subst_cddmatrix_ij(&ctmp2, a, j, k);
				rcdd_sub(&ctmp, &ctmp2, &ctmp1);
				set_cddmatrix_ij(a, j, k, &ctmp);
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
int SolveCDDLSPM(CDDVector answer, CDDMatrix lu, CDDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*     CDDMatrix lu[]: LU decomposed Matrix (given by user) */
/*     CDDVector b[]: constant vector (given by user)       */
/*     CDDVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[DDSIZE], dtmp1[DDSIZE];
    static cddfloat ctmp, ctmp1;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_cddvector_i(answer, i, get_cddvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcdd_abs_dd(dtmp, get_cddmatrix_ij(lu, i, i));
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCDDLSPM, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcdd_mul(&ctmp1, get_cddmatrix_ij(lu, j, i), get_cddvector_i(answer, i));
			rcdd_sub(&ctmp, get_cddvector_i(answer, j), &ctmp1);
			set_cddvector_i(answer, j, &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcdd_mul(&ctmp1, get_cddmatrix_ij(lu, i, j), get_cddvector_i(answer, j));
			rcdd_sub(&ctmp, get_cddvector_i(answer, i), &ctmp1);
			set_cddvector_i(answer, i, &ctmp);
		}
		//rcdd_div(&ctmp, get_cddvector_i(answer, i), get_cddmatrix_ij(lu, i, i));
		rcdd_inv(&ctmp1, get_cddmatrix_ij(lu, i, i));
		rcdd_mul(&ctmp, get_cddvector_i(answer, i), &ctmp1);
		set_cddvector_i(answer, i, &ctmp);
	}

	return 0;
}
