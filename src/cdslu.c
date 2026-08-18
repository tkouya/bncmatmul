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
//#include "dslinear.h"
#include "cdslinear.h"

// DD

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                 (Complex Double-Double Precision)        */
/*                                                          */
/*                 Ver. 0.0 2023-12-11 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CDSLUdecomp(CDSMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CDSMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	//static 
	cdsfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;
	//static 
	float dmaxii[DSSIZE];

	//dim = a->col_dim;
	dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
	{
		//rcds_abs_ds(dmaxii, get_cdsmatrix_ij(a, i, i));
		subst_cdsmatrix_ij(&aii, a, i, i);
		rcds_abs_ds(dmaxii, &aii); // get_cdsmatrix_ij(a, i, i));
		//printf("a%ld_%ld = ", i, i); rds_out_str_base(stdout, 10, 33, dmaxii); printf("\n");
		if(rds_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CDSLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			//rcds_nrm2_ds(dmaxii, get_cdsmatrix_ij(a, i, i)); rds_out_str_base(stdout, 10, 33, dmaxii); printf("\n");
			//rcds_conj(&ctmp, get_cdsmatrix_ij(a, j, i)); rcds_out_str(&ctmp); printf("\n");
			subst_cdsmatrix_ij(&aii, a, i, i); //rcds_out_str(&ctmp1); printf("\n");	
			rcds_inv(&ctmp1, &aii); // get_cdsmatrix_ij(a, i, i)); //rcds_out_str(&ctmp1); printf("\n");
			subst_cdsmatrix_ij(&aji, a, j, i);
			rcds_mul_3m(&ctmp, &aji, &ctmp1); //rcds_out_str(&ctmp); printf("\n");
			//rcds_div_3m(&ctmp, get_cdsmatrix_ij(a, j, i), get_cdsmatrix_ij(a, i, i));
			//rcds_out_str(&ctmp); printf("\n");
			set_cdsmatrix_ij(a, j, i, &ctmp);
			//printf("a%ld_%ld = ", j, i); rcds_out_str(&ctmp); printf("\n");
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				//rcds_mul(&ctmp1, get_cdsmatrix_ij(a, j, i), get_cdsmatrix_ij(a, i, k));
				//rcds_mul_3m(&ctmp1, get_cdsmatrix_ij(a, j, i), get_cdsmatrix_ij(a, i, k));
				subst_cdsmatrix_ij(&aji, a, j, i);
				subst_cdsmatrix_ij(&aik, a, i, k);
				rcds_mul_3m(&ctmp1, &aji, &aik); // get_cdsmatrix_ij(a, j, i), get_cdsmatrix_ij(a, i, k));
				//printf("a%ld_%ld= ", j, k); rcds_out_str(&ctmp1); printf("\n");
				subst_cdsmatrix_ij(&ajk, a, j, k);
				//rcds_sub(&ctmp, get_cdsmatrix_ij(a, j, k), &ctmp1);
				rcds_sub(&ctmp, &ajk, &ctmp1);
				set_cdsmatrix_ij(a, j, k, &ctmp);
				//printf("a%ld_%ld= ", j, k); rcds_out_str(&ctmp); printf("\n");
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                      (Complex Double-float Precision)   */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCDSLS(CDSVector answer, CDSMatrix lu, CDSVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CDSMatrix lu: LU decomposed Matrix (given by user)  */
/*      CDSVector b: constant vector (given by user)        */
/*      CDSVector answer: Solution for linear system        */
/*      long int dim: Dimension of Matrix (given by user)   */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	//static 
	cdsfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;
    //static 
	float dtmp[DSSIZE];

	dim = answer->re->dim;

	subst_cdsvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		//printf("f %ld = ", i); rcds_out_str(get_cdsvector_i(answer, i)); printf("\n");
		rcds_abs_ds(dtmp, get_cdsmatrix_ij(lu, i, i));
		if(rds_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCDSLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			//rcds_mul(&ctmp1, get_cdsmatrix_ij(lu, j, i), get_cdsvector_i(answer, i));
			rcds_mul_3m(&ctmp1, get_cdsmatrix_ij(lu, j, i), get_cdsvector_i(answer, i));
			rcds_sub(&ctmp, get_cdsvector_i(answer, j), &ctmp1);
			set_cdsvector_i(answer, j, &ctmp);
		}
		//printf("f %ld = ", i); rcds_out_str(get_cdsvector_i(answer, i)); printf("\n");
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			//rcds_mul(&ctmp1, get_cdsmatrix_ij(lu, i, j), get_cdsvector_i(answer, j));
			rcds_mul_3m(&ctmp1, get_cdsmatrix_ij(lu, i, j), get_cdsvector_i(answer, j));
			rcds_sub(&ctmp, get_cdsvector_i(answer, i), &ctmp1);
			set_cdsvector_i(answer, i, &ctmp);
		}
		//rcds_div(&ctmp, get_cdsvector_i(answer, i), get_cdsmatrix_ij(lu, i, i));
		rcds_inv(&ctmp1, get_cdsmatrix_ij(lu, i, i));
		//rcds_mul(&ctmp, get_cdsvector_i(answer, i), &ctmp1);
		rcds_mul_3m(&ctmp, get_cdsvector_i(answer, i), &ctmp1);
		set_cdsvector_i(answer, i, &ctmp);
		//printf("b %ld = ", i); rcds_out_str(get_cdsvector_i(answer, i)); printf("\n");
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
int CDSLUdecompP(CDSMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DSMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	static cdsfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;
    static float dmaxii[DSSIZE], dtmp[DSSIZE];

	//dim = a->col_dim;
    dim = a->re->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		rcds_abs_ds(dmaxii, get_cdsmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rcds_abs_ds(dtmp, get_cdsmatrix_ij(a, ch[j], i));
			if(rds_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rds_set(dmaxii, dtmp);
			}
		}

		if(rds_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CDSLUdecompP!\n", i);
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
			//rcds_div(&ctmp, get_cdsmatrix_ij(a, ch[j], i), get_cdsmatrix_ij(a, ch[i], i));
			rcds_inv(&ctmp1, get_cdsmatrix_ij(a, ch[i], i));
			rcds_mul(&ctmp, get_cdsmatrix_ij(a, ch[j], i), &ctmp1);
			set_cdsmatrix_ij(a, ch[j], i, &ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcds_mul(&ctmp1, get_cdsmatrix_ij(a, ch[j], i), get_cdsmatrix_ij(a, ch[i], k));
				rcds_sub(&ctmp, get_cdsmatrix_ij(a, ch[j], k), &ctmp1);
				set_cdsmatrix_ij(a, ch[j], k, &ctmp);
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
int SolveCDSLSP(CDSVector answer, CDSMatrix lu, CDSVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*     CDSMatrix lu[]: LU decomposed Matrix (given by user) */
/*     CDSVector b[]: constant vector (given by user)       */
/*     CDSVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static float dtmp[DSSIZE], dtmp1[DSSIZE];
    static cdsfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_cdsvector_i(answer, i, get_cdsvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcds_abs_ds(dtmp, get_cdsmatrix_ij(lu, ch[i], i));
		if(rds_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCDSLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcds_mul(&ctmp1, get_cdsmatrix_ij(lu, ch[j], i), get_cdsvector_i(answer, i));
			rcds_sub(&ctmp, get_cdsvector_i(answer, j), &ctmp1);
			set_cdsvector_i(answer, j, &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcds_mul(&ctmp1, get_cdsmatrix_ij(lu, ch[i], j), get_cdsvector_i(answer, j));
			rcds_sub(&ctmp, get_cdsvector_i(answer, i), &ctmp1);
			set_cdsvector_i(answer, i, &ctmp);
		}
		//rcds_div(&ctmp, get_cdsvector_i(answer, i), get_cdsmatrix_ij(lu, ch[i], i));
		rcds_inv(&ctmp1, get_cdsmatrix_ij(lu, ch[i], i));
		rcds_mul(&ctmp, get_cdsvector_i(answer, i), &ctmp1);
		set_cdsvector_i(answer, i, &ctmp);
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
int CDSLUdecompC(CDSMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDSMatrix a[]: Matrix (given by user)              */
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
	static float dtmp[DSSIZE], dtmp1[DSSIZE], dmaxii[DSSIZE];
    static cdsfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

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
		rcds_abs_ds(dmaxii, get_cdsmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcds_abs_ds(dtmp, get_cdsmatrix_ij(a, row_ch[j], col_ch[k]));
				if(rds_cmp(dtmp, dmaxii) > 0)
				{
					imax = j;
					jmax = k;
					rds_set(dmaxii, dtmp);
				}
			}
		}

		if(rds_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CDSLUdecompC)!\n", i);
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
			//rcds_div(&ctmp, get_cdsmatrix_ij(a, row_ch[j], col_ch[i]), get_cdsmatrix_ij(a, row_ch[i], col_ch[i]));
			rcds_inv(&ctmp1, get_cdsmatrix_ij(a, row_ch[i], col_ch[i]));
			rcds_mul(&ctmp, get_cdsmatrix_ij(a, row_ch[j], col_ch[i]), &ctmp1);
			set_cdsmatrix_ij(a, row_ch[j], col_ch[i], &ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rcds_mul(&ctmp1, get_cdsmatrix_ij(a, row_ch[j], col_ch[i]), get_cdsmatrix_ij(a, row_ch[i], col_ch[k]));
				rcds_sub(&ctmp, get_cdsmatrix_ij(a, row_ch[j], col_ch[k]), &ctmp1);
				set_cdsmatrix_ij(a, row_ch[j], col_ch[k], &ctmp);
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
int SolveCDSLSC(CDSVector answer, CDSMatrix lu, CDSVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDSMatrix lu: LU decomposed Matrix (given by user) */
/*       CDSVector b: constant vector (given by user)       */
/*       CDSVector answer: Solution for linear system       */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static float dtmp[DSSIZE], dtmp1[DSSIZE];
    static cdsfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_cdsvector_i(answer, col_ch[i], get_cdsvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcds_abs_ds(dtmp, get_cdsmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rds_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCDSLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcds_mul(&ctmp1, get_cdsmatrix_ij(lu, row_ch[j], col_ch[i]), get_cdsvector_i(answer, col_ch[i]));
			rcds_sub(&ctmp, get_cdsvector_i(answer, col_ch[j]), &ctmp1);
			set_cdsvector_i(answer, col_ch[j], &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcds_mul(&ctmp1, get_cdsmatrix_ij(lu, row_ch[i], col_ch[j]), get_cdsvector_i(answer, col_ch[j]));
			rcds_sub(&ctmp, get_cdsvector_i(answer, col_ch[i]), &ctmp1);
			set_cdsvector_i(answer, col_ch[i], &ctmp);
		}
		//rcds_div(&ctmp, get_cdsvector_i(answer, col_ch[i]), get_cdsmatrix_ij(lu, row_ch[i], col_ch[i]));
		rcds_inv(&ctmp1, get_cdsmatrix_ij(lu, row_ch[i], col_ch[i]));
		rcds_mul(&ctmp, get_cdsvector_i(answer, col_ch[i]), &ctmp1);
		set_cdsvector_i(answer, col_ch[i], &ctmp);
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                      (Complex Double float Precision)   */
/*              (Partial Pivoting with real swap of rows)   */
/*                                                          */
/*                 Ver. 0.0 2023-12-11 (Mon) Tomonori Kouya */
/*                 Ver. 0.1 2025-12-19 (Thu) Neon/AVX512    */
/*                                                          */
/************************************************************/
int CDSLUdecompPM(CDSMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDSMatrix a: Matrix (given by user)                */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim, row_dim;
	static float dtmp[DSSIZE], dtmp1[DSSIZE], dmaxii[DSSIZE];
    static cdsfloat ctmp, ctmp1, ctmp2, aii, aji, aik, ajk;
// SIMD : for copy & paste
#if 0 // __AVX2__ (disabled for single-complex)
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256d ctmp256_re[DSSIZE], aji256_re[DSSIZE], ajk256_re[DSSIZE], aik256_re[DSSIZE];
	__m256d ctmp256_im[DSSIZE], aji256_im[DSSIZE], ajk256_im[DSSIZE], aik256_im[DSSIZE];
#elif 0 // __AVX512F__ (disabled)
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m512d ctmp512_re[DSSIZE], aji512_re[DSSIZE], ajk512_re[DSSIZE], aik512_re[DSSIZE];
	__m512d ctmp512_im[DSSIZE], aji512_im[DSSIZE], ajk512_im[DSSIZE], aik512_im[DSSIZE];
#elif 0 // Arm Neon (disabled for single-complex)
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	float64x2_t ctmp_neon_re[DSSIZE], aji_neon_re[DSSIZE], ajk_neon_re[DSSIZE], aik_neon_re[DSSIZE];
	float64x2_t ctmp_neon_im[DSSIZE], aji_neon_im[DSSIZE], ajk_neon_im[DSSIZE], aik_neon_im[DSSIZE];
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
		//printf("i = %ld -> ", i); rcds_out_str(get_cdsmatrix_ij(a, i, i)); printf("\n");
		subst_cdsmatrix_ij(&aii, a, i, i);
		//printf("i = %ld -> ", i); rcds_out_str(&ctmp); printf("\n");
		//rcds_abs_ds(dmaxii, get_cdsmatrix_ij(a, i, i));
		rcds_abs_ds(dmaxii, &aii);
		//printf("dmaxii = %25.17e\n", dmaxii[0]);
		imax = i;
		for(j = (i + 1); j < row_dim; j++)
		{
			//rcds_abs_ds(dtmp, get_cdsmatrix_ij(a, j, i));
			subst_cdsmatrix_ij(&aji, a, j, i);
			rcds_abs_ds(dtmp, &aji);
			if(rds_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rds_set(dmaxii, dtmp);
			}
		}
		//printf("dmaxii = %25.17e, %ld -> %ld\n", dmaxii[0], i, imax);
		if(rds_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CDSLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_cdsmatrix(a, i, imax, 0, a->re->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			//rcds_div(&ctmp, get_cdsmatrix_ij(a, j, i), get_cdsmatrix_ij(a, i, i));
			subst_cdsmatrix_ij(&aii, a, i, i);
			//rcds_inv(&ctmp1, get_cdsmatrix_ij(a, i, i));
			rcds_inv(&ctmp1, &aii); //get_cdsmatrix_ij(a, i, i));
			subst_cdsmatrix_ij(&aji, a, j, i);
			//rcds_mul(&ctmp, get_cdsmatrix_ij(a, j, i), &ctmp1);
			rcds_mul(&ctmp, &aji, &ctmp1);
			set_cdsmatrix_ij(a, j, i, &ctmp);
		}
// SIMD : for copy & paste
#if 0 // __AVX2__ (disabled for single-complex)
		dim_start = (long int)ceil((float)(i + 1) / (float)_BNC_D_WIDTH) * _BNC_D_WIDTH;
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
				//rcds_mul(&ctmp1, get_cdsmatrix_ij(a, j, i), get_cdsmatrix_ij(a, i, k));
				//rcds_sub(&ctmp, get_cdsmatrix_ij(a, j, k), &ctmp1);
				subst_cdsmatrix_ij(&aji, a, j, i);
				subst_cdsmatrix_ij(&aik, a, i, k);
				rcds_mul(&ctmp1, &aji, &aik);

				subst_cdsmatrix_ij(&ajk, a, j, k);
				rcds_sub(&ctmp, &ajk, &ctmp1);
				set_cdsmatrix_ij(a, j, k, &ctmp);
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
				aik256_im[0] = _mm256_load_pd(&(a->im->element[0][index_ik]));
				aik256_im[1] = _mm256_load_pd(&(a->im->element[1][index_ik]));
				_bncavx2_rcds_mul(ctmp256_re, ctmp256_im, aji256_re, aji256_im, aik256_re, aik256_im);
				//printf(" -- mul -- ");

				index_jk = j * a->re->real_col_dim + k;
				//rcds_sub(dtmp, get_dsmatrix_ij(a, j, k), dtmp1);
				ajk256_re[0] = _mm256_load_pd(&(a->re->element[0][index_jk]));
				ajk256_re[1] = _mm256_load_pd(&(a->re->element[1][index_jk]));
				ajk256_im[0] = _mm256_load_pd(&(a->im->element[0][index_jk]));
				ajk256_im[1] = _mm256_load_pd(&(a->im->element[1][index_jk]));
				_bncavx2_rcds_sub(ctmp256_re, ctmp256_im, ajk256_re, ajk256_im, ctmp256_re, ctmp256_im);
				//printf(" -- sub -- ");

				//set_cdsmatrix_ij(a, j, k, dtmp);
				_mm256_store_pd(&(a->re->element[0][index_jk]), ctmp256_re[0]);
				_mm256_store_pd(&(a->re->element[1][index_jk]), ctmp256_re[1]);
				_mm256_store_pd(&(a->im->element[0][index_jk]), ctmp256_im[0]);
				_mm256_store_pd(&(a->im->element[1][index_jk]), ctmp256_im[1]);
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
			aji512_re[0] = _mm512_set1_pd(a->re->element[0][index_ji]);
			aji512_re[1] = _mm512_set1_pd(a->re->element[1][index_ji]);
			aji512_im[0] = _mm512_set1_pd(a->im->element[0][index_ji]);
			aji512_im[1] = _mm512_set1_pd(a->im->element[1][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				subst_cdsmatrix_ij(&aji, a, j, i);
				subst_cdsmatrix_ij(&aik, a, i, k);
				rcds_mul(&ctmp1, &aji, &aik);

				subst_cdsmatrix_ij(&ajk, a, j, k);
				rcds_sub(&ctmp, &ajk, &ctmp1);
				set_cdsmatrix_ij(a, j, k, &ctmp);
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
				_bncavx512_rcds_mul(ctmp512_re, ctmp512_im, aji512_re, aji512_im, aik512_re, aik512_im);

				index_jk = j * a->re->real_col_dim + k;
				// Load a[j][k] elements (real and imaginary parts)
				ajk512_re[0] = _mm512_load_pd(&(a->re->element[0][index_jk]));
				ajk512_re[1] = _mm512_load_pd(&(a->re->element[1][index_jk]));
				ajk512_im[0] = _mm512_load_pd(&(a->im->element[0][index_jk]));
				ajk512_im[1] = _mm512_load_pd(&(a->im->element[1][index_jk]));
				
				// Complex subtract: ctmp = a[j][k] - ctmp
				_bncavx512_rcds_sub(ctmp512_re, ctmp512_im, ajk512_re, ajk512_im, ctmp512_re, ctmp512_im);

				// Store result back to a[j][k] (real and imaginary parts)
				_mm512_store_pd(&(a->re->element[0][index_jk]), ctmp512_re[0]);
				_mm512_store_pd(&(a->re->element[1][index_jk]), ctmp512_re[1]);
				_mm512_store_pd(&(a->im->element[0][index_jk]), ctmp512_im[0]);
				_mm512_store_pd(&(a->im->element[1][index_jk]), ctmp512_im[1]);
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
			aji_neon_re[0] = vdupq_n_f64(a->re->element[0][index_ji]);
			aji_neon_re[1] = vdupq_n_f64(a->re->element[1][index_ji]);
			aji_neon_im[0] = vdupq_n_f64(a->im->element[0][index_ji]);
			aji_neon_im[1] = vdupq_n_f64(a->im->element[1][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				subst_cdsmatrix_ij(&aji, a, j, i);
				subst_cdsmatrix_ij(&aik, a, i, k);
				rcds_mul(&ctmp1, &aji, &aik);

				subst_cdsmatrix_ij(&ajk, a, j, k);
				rcds_sub(&ctmp, &ajk, &ctmp1);
				set_cdsmatrix_ij(a, j, k, &ctmp);
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
				_bncneon_rcds_mul(ctmp_neon_re, ctmp_neon_im, aji_neon_re, aji_neon_im, aik_neon_re, aik_neon_im);

				index_jk = j * a->re->real_col_dim + k;
				// Load a[j][k] elements (real and imaginary parts)
				ajk_neon_re[0] = vld1q_f64(&(a->re->element[0][index_jk]));
				ajk_neon_re[1] = vld1q_f64(&(a->re->element[1][index_jk]));
				ajk_neon_im[0] = vld1q_f64(&(a->im->element[0][index_jk]));
				ajk_neon_im[1] = vld1q_f64(&(a->im->element[1][index_jk]));
				
				// Complex subtract: ctmp = a[j][k] - ctmp
				_bncneon_rcds_sub(ctmp_neon_re, ctmp_neon_im, ajk_neon_re, ajk_neon_im, ctmp_neon_re, ctmp_neon_im);

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
				subst_cdsmatrix_ij(&ctmp, a, j, i);
				subst_cdsmatrix_ij(&ctmp2, a, i, k);
				//rcds_mul(&ctmp1, get_cdsmatrix_ij(a, j, i), get_cdsmatrix_ij(a, i, k));
				rcds_mul(&ctmp1, &ctmp, &ctmp2); //get_cdsmatrix_ij(a, i, k));
				//rcds_sub(&ctmp, get_cdsmatrix_ij(a, j, k), &ctmp1);
				subst_cdsmatrix_ij(&ctmp2, a, j, k);
				rcds_sub(&ctmp, &ctmp2, &ctmp1);
				set_cdsmatrix_ij(a, j, k, &ctmp);
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
int SolveCDSLSPM(CDSVector answer, CDSMatrix lu, CDSVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*     CDSMatrix lu[]: LU decomposed Matrix (given by user) */
/*     CDSVector b[]: constant vector (given by user)       */
/*     CDSVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static float dtmp[DSSIZE], dtmp1[DSSIZE];
    static cdsfloat ctmp, ctmp1;

	dim = answer->re->dim;

	for(i = 0; i < dim; i++)
		set_cdsvector_i(answer, i, get_cdsvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rcds_abs_ds(dtmp, get_cdsmatrix_ij(lu, i, i));
		if(rds_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCDSLSPM, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rcds_mul(&ctmp1, get_cdsmatrix_ij(lu, j, i), get_cdsvector_i(answer, i));
			rcds_sub(&ctmp, get_cdsvector_i(answer, j), &ctmp1);
			set_cdsvector_i(answer, j, &ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rcds_mul(&ctmp1, get_cdsmatrix_ij(lu, i, j), get_cdsvector_i(answer, j));
			rcds_sub(&ctmp, get_cdsvector_i(answer, i), &ctmp1);
			set_cdsvector_i(answer, i, &ctmp);
		}
		//rcds_div(&ctmp, get_cdsvector_i(answer, i), get_cdsmatrix_ij(lu, i, i));
		rcds_inv(&ctmp1, get_cdsmatrix_ij(lu, i, i));
		rcds_mul(&ctmp, get_cdsvector_i(answer, i), &ctmp1);
		set_cdsvector_i(answer, i, &ctmp);
	}

	return 0;
}
