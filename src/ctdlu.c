/********************************************************************************/
/* ctdlu.c:                                                                      */
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
#include "ctdlinear.h"

// TD

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Complex Triple-double Precision)  */
/*                                                          */
/*                 Ver. 0.0 2024-01-31 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CTDLUdecomp(CTDMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CTDMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];
    ctdfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
	{
		rctd_abs_td(dmaxii, get_ctdmatrix_ij(a, i, i));
		printf("a%ld_%ld = ", i, i); rtd_out_str(dmaxii); printf("\n");
		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CTDLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			//rctd_div(&ctmp, get_ctdmatrix_ij(a, j, i), get_ctdmatrix_ij(a, i, i));
			rctd_inv(&ctmp1, get_ctdmatrix_ij(a, i, i));
			rctd_mul_3m(&ctmp, get_ctdmatrix_ij(a, j, i), &ctmp1);
			set_ctdmatrix_ij(a, j, i, &ctmp);
			//printf("a%ld_%ld = ", j, i); rctd_out_str(get_ctdmatrix_ij(a, j, i)); printf("\n");
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rctd_mul_3m(&ctmp1, get_ctdmatrix_ij(a, j, i), get_ctdmatrix_ij(a, i, k));
				//rctd_mul(&ctmp1, get_ctdmatrix_ij(a, j, i), get_ctdmatrix_ij(a, i, k));
				rctd_sub(&ctmp, get_ctdmatrix_ij(a, j, k), &ctmp1);
				set_ctdmatrix_ij(a, j, k, &ctmp);
				//printf("a%ld_%ld = ", j, k); rctd_out_str(&ctmp); printf("\n");
			}
		}
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                (LU Decomposision of Square Dense Matrix) */
/*                         (Complex Triple-double Precision)  */
/*                                                          */
/*                 Ver. 0.0 2024-01-31 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCTDLS(CTDVector answer, CTDMatrix lu, CTDVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CTDMatrix lu: LU decomposed Matrix (given by user)  */
/*      CTDVector b: constant vector (given by user)         */
/*      CTDVector answer: Solution for linear system         */
/*      long int dim: Dimension of Matrix (given by user)   */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp[TDSIZE], dtmp1[TDSIZE];
    ctdfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = answer->re->dim;

	subst_ctdvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rctd_abs_td(dtmp, get_ctdmatrix_ij(lu, i, i));
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCTDLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rctd_mul_3m(&ctmp1, get_ctdmatrix_ij(lu, j, i), get_ctdvector_i(answer, i));
			rctd_sub(&ctmp, get_ctdvector_i(answer, j), &ctmp1);
			set_ctdvector_i(answer, j, &ctmp);
		}
		//printf("f %ld = ", i); rctd_out_str(get_ctdvector_i(answer, i)); printf("\n");
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rctd_mul_3m(&ctmp1, get_ctdmatrix_ij(lu, i, j), get_ctdvector_i(answer, j));
			rctd_sub(&ctmp, get_ctdvector_i(answer, i), &ctmp1);
			set_ctdvector_i(answer, i, &ctmp);
		}
		//rctd_div(&ctmp, get_ctdvector_i(answer, i), get_ctdmatrix_ij(lu, i, i));
		rctd_inv(&ctmp1, get_ctdmatrix_ij(lu, i, i));
		rctd_mul_3m(&ctmp, get_ctdvector_i(answer, i), &ctmp1);
		set_ctdvector_i(answer, i, &ctmp);
		//printf("b %ld = ", i); rctd_out_str(get_ctdvector_i(answer, i)); printf("\n");
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Complex Triple-double Precision)  */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2024-01-31 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CTDLUdecompP(CTDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CTDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];
    ctdfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		rctd_abs_td(dmaxii, get_ctdmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rctd_abs_td(dtmp, get_ctdmatrix_ij(a, ch[j], i));
			if(rtd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rtd_set(dmaxii, dtmp);
			}
		}

		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CTDLUdecompP!\n", i);
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
			rctd_div(&ctmp, get_ctdmatrix_ij(a, ch[j], i), get_ctdmatrix_ij(a, ch[i], i));
			set_ctdmatrix_ij(a, ch[j], i, &ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rctd_mul(&ctmp1, get_ctdmatrix_ij(a, ch[j], i), get_ctdmatrix_ij(a, ch[i], k));
				rctd_sub(&ctmp, get_ctdmatrix_ij(a, ch[j], k), &ctmp1);
				set_ctdmatrix_ij(a, ch[j], k, &ctmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Complex Triple-double Precision)  */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2024-01-31 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCTDLSP(CTDVector answer, CTDMatrix lu, CTDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CTDMatrix lu[]: LU decomposed Matrix (given by user)*/
/*      CTDVector b[]: constant vector (given by user)      */
/*      CTDVector answer[]: Solution for linear system      */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp[TDSIZE], dtmp1[TDSIZE];
    ctdfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_ctdvector_i(answer, i, get_ctdvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rctd_abs_td(dtmp, get_ctdmatrix_ij(lu, ch[i], i));
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCTDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rctd_mul(&ctmp1, get_ctdmatrix_ij(lu, ch[j], i), get_ctdvector_i(answer, i));
			rctd_sub(&ctmp, get_ctdvector_i(answer, j), &ctmp1);
			set_ctdvector_i(answer, j, &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rctd_mul(&ctmp1, get_ctdmatrix_ij(lu, ch[i], j), get_ctdvector_i(answer, j));
			rctd_sub(&ctmp, get_ctdvector_i(answer, i), &ctmp1);
			set_ctdvector_i(answer, i, &ctmp);
		}
		rctd_div(&ctmp, get_ctdvector_i(answer, i), get_ctdmatrix_ij(lu, ch[i], i));
		set_ctdvector_i(answer, i, &ctmp);
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Complex Triple-double Precision)   */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CTDLUdecompC(CTDMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CTDMatrix a[]: Matrix (given by user)              */
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
	double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];
    ctdfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		rctd_abs_td(dmaxii, get_ctdmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rctd_abs_td(dtmp, get_ctdmatrix_ij(a, row_ch[j], col_ch[k]));
				if(rtd_cmp(dtmp, dmaxii) > 0)
				{
					imax = j;
					jmax = k;
					rtd_set(dmaxii, dtmp);
				}
			}
		}

		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CTDLUdecompC)!\n", i);
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
			rctd_div(&ctmp, get_ctdmatrix_ij(a, row_ch[j], col_ch[i]), get_ctdmatrix_ij(a, row_ch[i], col_ch[i]));
			set_ctdmatrix_ij(a, row_ch[j], col_ch[i], &ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rctd_mul(&ctmp1, get_ctdmatrix_ij(a, row_ch[j], col_ch[i]), get_ctdmatrix_ij(a, row_ch[i], col_ch[k]));
				rctd_sub(&ctmp, get_ctdmatrix_ij(a, row_ch[j], col_ch[k]), &ctmp1);
				set_ctdmatrix_ij(a, row_ch[j], col_ch[k], &ctmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Complex Triple-double Precision)  */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2024-01-31 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCTDLSC(CTDVector answer, CTDMatrix lu, CTDVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CTDMatrix lu: LU decomposed Matrix (given by user) */
/*       CTDVector b: constant vector (given by user)       */
/*       CTDVector answer: Solution for linear system       */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp[TDSIZE], dtmp1[TDSIZE];
    ctdfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_ctdvector_i(answer, col_ch[i], get_ctdvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rctd_abs_td(dtmp, get_ctdmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCTDLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rctd_mul(&ctmp1, get_ctdmatrix_ij(lu, row_ch[j], col_ch[i]), get_ctdvector_i(answer, col_ch[i]));
			rctd_sub(&ctmp, get_ctdvector_i(answer, col_ch[j]), &ctmp1);
			set_ctdvector_i(answer, col_ch[j], &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rctd_mul(&ctmp1, get_ctdmatrix_ij(lu, row_ch[i], col_ch[j]), get_ctdvector_i(answer, col_ch[j]));
			rctd_sub(&ctmp, get_ctdvector_i(answer, col_ch[i]), &ctmp1);
			set_ctdvector_i(answer, col_ch[i], &ctmp);
		}
		rctd_div(&ctmp, get_ctdvector_i(answer, col_ch[i]), get_ctdmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_ctdvector_i(answer, col_ch[i], &ctmp);
	}

	return 0;
}


/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                      (Complex Triple double Precision)   */
/*              (Partial Pivoting with real swap of rows)   */
/*                                                          */
/*                 Ver. 0.0 2024-01-31 (Wed) Tomonori Kouya */
/*                 Ver. 0.1 2025-12-19 (Thu) Neon/AVX512    */
/*                                                          */
/************************************************************/
int CTDLUdecompPM(CTDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CTDMatrix a: Matrix (given by user)                */
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
	double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];
    //
	ctdfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256d ctmp256_re[TDSIZE], aji256_re[TDSIZE], ajk256_re[TDSIZE], aik256_re[TDSIZE];
	__m256d ctmp256_im[TDSIZE], aji256_im[TDSIZE], ajk256_im[TDSIZE], aik256_im[TDSIZE];
#elif defined(__AVX512F__) // __AVX512F__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m512d ctmp512_re[TDSIZE], aji512_re[TDSIZE], ajk512_re[TDSIZE], aik512_re[TDSIZE];
	__m512d ctmp512_im[TDSIZE], aji512_im[TDSIZE], ajk512_im[TDSIZE], aik512_im[TDSIZE];
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	
	svfloat64_t ctmp_neon_re_0, ctmp_neon_re_1, ctmp_neon_re_2;
	svfloat64_t aji_neon_re_0, aji_neon_re_1, aji_neon_re_2;
	svfloat64_t ajk_neon_re_0, ajk_neon_re_1, ajk_neon_re_2;
	svfloat64_t aik_neon_re_0, aik_neon_re_1, aik_neon_re_2;
	
	svfloat64_t ctmp_neon_im_0, ctmp_neon_im_1, ctmp_neon_im_2;
	svfloat64_t aji_neon_im_0, aji_neon_im_1, aji_neon_im_2;
	svfloat64_t ajk_neon_im_0, ajk_neon_im_1, ajk_neon_im_2;
	svfloat64_t aik_neon_im_0, aik_neon_im_1, aik_neon_im_2;
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	float64x2_t ctmp_neon_re[TDSIZE], aji_neon_re[TDSIZE], ajk_neon_re[TDSIZE], aik_neon_re[TDSIZE];
	float64x2_t ctmp_neon_im[TDSIZE], aji_neon_im[TDSIZE], ajk_neon_im[TDSIZE], aik_neon_im[TDSIZE];
#else // normal
#endif // __AVX2__


#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	printf("CTDLUdecompPM AVX2 enabled!\n");
#elif defined(__AVX512F__) // __AVX512F__
	printf("CTDLUdecompPM AVX-512 enabled!\n");
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	printf("CTDLUdecompPM Arm Neon enabled!\n");
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	printf("CTDLUdecompPM Arm Neon enabled!\n");
#else
	printf("CTDLUdecompPM normal!\n");
#endif // __AVX2__

	//dim = a->col_dim;
    dim = a->re->col_dim;
	row_dim = a->re->row_dim;

	for(i = 0; i < row_dim; i++)
		ch[i] = i;

	for(i = 0; i < row_dim; i++)
	{
		// partial pivoting
		//rctd_abs_td(dmaxii, get_ctdmatrix_ij(a, i, i));
		subst_ctdmatrix_ij(&aii, a, i, i);
		rctd_abs_td(dmaxii, &aii); // get_ctdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < row_dim; j++)
		{
			//rctd_abs_td(dtmp, get_ctdmatrix_ij(a, j, i));
			subst_ctdmatrix_ij(&aji, a, j, i);
			rctd_abs_td(dtmp, &aji); // get_ctdmatrix_ij(a, j, i));
			if(rtd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rtd_set(dmaxii, dtmp);
			}
		}

		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CTDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_ctdmatrix(a, i, imax, 0, a->re->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			//rctd_div(&ctmp, get_ctdmatrix_ij(a, j, i), get_ctdmatrix_ij(a, i, i));
			subst_ctdmatrix_ij(&aii, a, i, i);
			rctd_inv(&ctmp1, &aii); // get_ctdmatrix_ij(a, i, i));
			//rctd_mul(&ctmp, get_ctdmatrix_ij(a, j, i), &ctmp1);
			subst_ctdmatrix_ij(&aji, a, j, i);
			rctd_mul(&ctmp, &aji, &ctmp1);
			set_ctdmatrix_ij(a, j, i, &ctmp);
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

			// head
			//printf("start j, k= %ld, %ld, ", j, i + 1);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			//for(k = (i + 1); k < dim; k++)
			{
				//rctd_mul(&ctmp1, get_ctdmatrix_ij(a, j, i), get_ctdmatrix_ij(a, i, k));
				subst_ctdmatrix_ij(&aji, a, j, i);
				subst_ctdmatrix_ij(&aik, a, i, k);
				rctd_mul(&ctmp1, &aji, &aik); // get_ctdmatrix_ij(a, j, i), get_ctdmatrix_ij(a, i, k));
				//rctd_sub(&ctmp, get_ctdmatrix_ij(a, j, k), &ctmp1);
				subst_ctdmatrix_ij(&ajk, a, j, k);
				rctd_sub(&ctmp, &ajk, &ctmp1);
				set_ctdmatrix_ij(a, j, k, &ctmp);
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

				aik256_im[0] = _mm256_load_pd(&(a->im->element[0][index_ik]));
				aik256_im[1] = _mm256_load_pd(&(a->im->element[1][index_ik]));
				aik256_im[2] = _mm256_load_pd(&(a->im->element[2][index_ik]));

				_bncavx2_rctd_mul(ctmp256_re, ctmp256_im, aji256_re, aji256_im, aik256_re, aik256_im);
				//printf(" -- mul -- ");

				index_jk = j * a->re->real_col_dim + k;
				//rcdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				ajk256_re[0] = _mm256_load_pd(&(a->re->element[0][index_jk]));
				ajk256_re[1] = _mm256_load_pd(&(a->re->element[1][index_jk]));
				ajk256_re[2] = _mm256_load_pd(&(a->re->element[2][index_jk]));

				ajk256_im[0] = _mm256_load_pd(&(a->im->element[0][index_jk]));
				ajk256_im[1] = _mm256_load_pd(&(a->im->element[1][index_jk]));
				ajk256_im[2] = _mm256_load_pd(&(a->im->element[2][index_jk]));
				_bncavx2_rctd_sub(ctmp256_re, ctmp256_im, ajk256_re, ajk256_im, ctmp256_re, ctmp256_im);
				//printf(" -- sub -- ");

				//set_cddmatrix_ij(a, j, k, dtmp);
				_mm256_store_pd(&(a->re->element[0][index_jk]), ctmp256_re[0]);
				_mm256_store_pd(&(a->re->element[1][index_jk]), ctmp256_re[1]);
				_mm256_store_pd(&(a->re->element[2][index_jk]), ctmp256_re[2]);

				_mm256_store_pd(&(a->im->element[0][index_jk]), ctmp256_im[0]);
				_mm256_store_pd(&(a->im->element[1][index_jk]), ctmp256_im[1]);
				_mm256_store_pd(&(a->im->element[2][index_jk]), ctmp256_im[2]);

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

			// Imaginary part
			aji512_im[0] = _mm512_set1_pd(a->im->element[0][index_ji]);
			aji512_im[1] = _mm512_set1_pd(a->im->element[1][index_ji]);
			aji512_im[2] = _mm512_set1_pd(a->im->element[2][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				subst_ctdmatrix_ij(&aji, a, j, i);
				subst_ctdmatrix_ij(&aik, a, i, k);
				rctd_mul(&ctmp1, &aji, &aik);

				subst_ctdmatrix_ij(&ajk, a, j, k);
				rctd_sub(&ctmp, &ajk, &ctmp1);
				set_ctdmatrix_ij(a, j, k, &ctmp);
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

				aik512_im[0] = _mm512_load_pd(&(a->im->element[0][index_ik]));
				aik512_im[1] = _mm512_load_pd(&(a->im->element[1][index_ik]));
				aik512_im[2] = _mm512_load_pd(&(a->im->element[2][index_ik]));
				
				// Complex multiply: ctmp = a[j][i] * a[i][k]
				_bncavx512_rctd_mul(ctmp512_re, ctmp512_im, aji512_re, aji512_im, aik512_re, aik512_im);

				index_jk = j * a->re->real_col_dim + k;
				// Load a[j][k] elements (real and imaginary parts)
				ajk512_re[0] = _mm512_load_pd(&(a->re->element[0][index_jk]));
				ajk512_re[1] = _mm512_load_pd(&(a->re->element[1][index_jk]));
				ajk512_re[2] = _mm512_load_pd(&(a->re->element[2][index_jk]));

				ajk512_im[0] = _mm512_load_pd(&(a->im->element[0][index_jk]));
				ajk512_im[1] = _mm512_load_pd(&(a->im->element[1][index_jk]));
				ajk512_im[2] = _mm512_load_pd(&(a->im->element[2][index_jk]));
				
				// Complex subtract: ctmp = a[j][k] - ctmp
				_bncavx512_rctd_sub(ctmp512_re, ctmp512_im, ajk512_re, ajk512_im, ctmp512_re, ctmp512_im);

				// Store result back to a[j][k] (real and imaginary parts)
				_mm512_store_pd(&(a->re->element[0][index_jk]), ctmp512_re[0]);
				_mm512_store_pd(&(a->re->element[1][index_jk]), ctmp512_re[1]);
				_mm512_store_pd(&(a->re->element[2][index_jk]), ctmp512_re[2]);

				_mm512_store_pd(&(a->im->element[0][index_jk]), ctmp512_im[0]);
				_mm512_store_pd(&(a->im->element[1][index_jk]), ctmp512_im[1]);
				_mm512_store_pd(&(a->im->element[2][index_jk]), ctmp512_im[2]);
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

			// Imaginary part
			aji_neon_im_0 = svdup_f64(a->im->element[0][index_ji]);
			aji_neon_im_1 = svdup_f64(a->im->element[1][index_ji]);
			aji_neon_im_2 = svdup_f64(a->im->element[2][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				subst_ctdmatrix_ij(&aji, a, j, i);
				subst_ctdmatrix_ij(&aik, a, i, k);
				rctd_mul(&ctmp1, &aji, &aik);

				subst_ctdmatrix_ij(&ajk, a, j, k);
				rctd_sub(&ctmp, &ajk, &ctmp1);
				set_ctdmatrix_ij(a, j, k, &ctmp);
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

				aik_neon_im_0 = svld1_f64(pg, &(a->im->element[0][index_ik]));
				aik_neon_im_1 = svld1_f64(pg, &(a->im->element[1][index_ik]));
				aik_neon_im_2 = svld1_f64(pg, &(a->im->element[2][index_ik]));
				
				// Complex multiply: ctmp = a[j][i] * a[i][k]
				_bncsve2_rctd_mul(pg, &ctmp_neon_re_0, &ctmp_neon_re_1, &ctmp_neon_re_2, &ctmp_neon_im_0, &ctmp_neon_im_1, &ctmp_neon_im_2, aji_neon_re_0, aji_neon_re_1, aji_neon_re_2, aji_neon_im_0, aji_neon_im_1, aji_neon_im_2, aik_neon_re_0, aik_neon_re_1, aik_neon_re_2, aik_neon_im_0, aik_neon_im_1, aik_neon_im_2);

				index_jk = j * a->re->real_col_dim + k;
				// Load a[j][k] elements (real and imaginary parts)
				ajk_neon_re_0 = svld1_f64(pg, &(a->re->element[0][index_jk]));
				ajk_neon_re_1 = svld1_f64(pg, &(a->re->element[1][index_jk]));
				ajk_neon_re_2 = svld1_f64(pg, &(a->re->element[2][index_jk]));

				ajk_neon_im_0 = svld1_f64(pg, &(a->im->element[0][index_jk]));
				ajk_neon_im_1 = svld1_f64(pg, &(a->im->element[1][index_jk]));
				ajk_neon_im_2 = svld1_f64(pg, &(a->im->element[2][index_jk]));
				
				// Complex subtract: ctmp = a[j][k] - ctmp
				_bncsve2_rctd_sub(pg, &ctmp_neon_re_0, &ctmp_neon_re_1, &ctmp_neon_re_2, &ctmp_neon_im_0, &ctmp_neon_im_1, &ctmp_neon_im_2, ajk_neon_re_0, ajk_neon_re_1, ajk_neon_re_2, ajk_neon_im_0, ajk_neon_im_1, ajk_neon_im_2, ctmp_neon_re_0, ctmp_neon_re_1, ctmp_neon_re_2, ctmp_neon_im_0, ctmp_neon_im_1, ctmp_neon_im_2);

				// Store result back to a[j][k] (real and imaginary parts)
				svst1_f64(pg, &(a->re->element[0][index_jk]), ctmp_neon_re_0);
				svst1_f64(pg, &(a->re->element[1][index_jk]), ctmp_neon_re_1);
				svst1_f64(pg, &(a->re->element[2][index_jk]), ctmp_neon_re_2);

				svst1_f64(pg, &(a->im->element[0][index_jk]), ctmp_neon_im_0);
				svst1_f64(pg, &(a->im->element[1][index_jk]), ctmp_neon_im_1);
				svst1_f64(pg, &(a->im->element[2][index_jk]), ctmp_neon_im_2);
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

			// Imaginary part
			aji_neon_im[0] = vdupq_n_f64(a->im->element[0][index_ji]);
			aji_neon_im[1] = vdupq_n_f64(a->im->element[1][index_ji]);
			aji_neon_im[2] = vdupq_n_f64(a->im->element[2][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				subst_ctdmatrix_ij(&aji, a, j, i);
				subst_ctdmatrix_ij(&aik, a, i, k);
				rctd_mul(&ctmp1, &aji, &aik);

				subst_ctdmatrix_ij(&ajk, a, j, k);
				rctd_sub(&ctmp, &ajk, &ctmp1);
				set_ctdmatrix_ij(a, j, k, &ctmp);
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

				aik_neon_im[0] = vld1q_f64(&(a->im->element[0][index_ik]));
				aik_neon_im[1] = vld1q_f64(&(a->im->element[1][index_ik]));
				aik_neon_im[2] = vld1q_f64(&(a->im->element[2][index_ik]));
				
				// Complex multiply: ctmp = a[j][i] * a[i][k]
				_bncneon_rctd_mul(ctmp_neon_re, ctmp_neon_im, aji_neon_re, aji_neon_im, aik_neon_re, aik_neon_im);

				index_jk = j * a->re->real_col_dim + k;
				// Load a[j][k] elements (real and imaginary parts)
				ajk_neon_re[0] = vld1q_f64(&(a->re->element[0][index_jk]));
				ajk_neon_re[1] = vld1q_f64(&(a->re->element[1][index_jk]));
				ajk_neon_re[2] = vld1q_f64(&(a->re->element[2][index_jk]));

				ajk_neon_im[0] = vld1q_f64(&(a->im->element[0][index_jk]));
				ajk_neon_im[1] = vld1q_f64(&(a->im->element[1][index_jk]));
				ajk_neon_im[2] = vld1q_f64(&(a->im->element[2][index_jk]));
				
				// Complex subtract: ctmp = a[j][k] - ctmp
				_bncneon_rctd_sub(ctmp_neon_re, ctmp_neon_im, ajk_neon_re, ajk_neon_im, ctmp_neon_re, ctmp_neon_im);

				// Store result back to a[j][k] (real and imaginary parts)
				vst1q_f64(&(a->re->element[0][index_jk]), ctmp_neon_re[0]);
				vst1q_f64(&(a->re->element[1][index_jk]), ctmp_neon_re[1]);
				vst1q_f64(&(a->re->element[2][index_jk]), ctmp_neon_re[2]);

				vst1q_f64(&(a->im->element[0][index_jk]), ctmp_neon_im[0]);
				vst1q_f64(&(a->im->element[1][index_jk]), ctmp_neon_im[1]);
				vst1q_f64(&(a->im->element[2][index_jk]), ctmp_neon_im[2]);
			}
		}
#else // others
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				//rctd_mul(&ctmp1, get_ctdmatrix_ij(a, j, i), get_ctdmatrix_ij(a, i, k));
				subst_ctdmatrix_ij(&aji, a, j, i);
				subst_ctdmatrix_ij(&aik, a, i, k);
				rctd_mul(&ctmp1, &aji, &aik);
				//rctd_sub(&ctmp, get_ctdmatrix_ij(a, j, k), &ctmp1);
				subst_ctdmatrix_ij(&ajk, a, j, k);
				rctd_sub(&ctmp, &ajk, &ctmp1);
				set_ctdmatrix_ij(a, j, k, &ctmp);
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
int SolveCTDLSPM(CTDVector answer, CTDMatrix lu, CTDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*     CTDMatrix lu[]: LU decomposed Matrix (given by user) */
/*     CTDVector b[]: constant vector (given by user)       */
/*     CTDVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp[TDSIZE], dtmp1[TDSIZE];
    ctdfloat ctmp, ctmp1;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_ctdvector_i(answer, i, get_ctdvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rctd_abs_td(dtmp, get_ctdmatrix_ij(lu, i, i));
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCTDLSPM, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rctd_mul(&ctmp1, get_ctdmatrix_ij(lu, j, i), get_ctdvector_i(answer, i));
			rctd_sub(&ctmp, get_ctdvector_i(answer, j), &ctmp1);
			set_ctdvector_i(answer, j, &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rctd_mul(&ctmp1, get_ctdmatrix_ij(lu, i, j), get_ctdvector_i(answer, j));
			rctd_sub(&ctmp, get_ctdvector_i(answer, i), &ctmp1);
			set_ctdvector_i(answer, i, &ctmp);
		}
		//rctd_div(&ctmp, get_ctdvector_i(answer, i), get_ctdmatrix_ij(lu, i, i));
		rctd_inv(&ctmp1, get_ctdmatrix_ij(lu, i, i));
		rctd_mul(&ctmp, get_ctdvector_i(answer, i), &ctmp1);
		set_ctdvector_i(answer, i, &ctmp);
	}

	return 0;
}