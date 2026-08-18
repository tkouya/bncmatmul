/********************************************************************************/
/* serial_lu_bench.c:                                                           */
/* Copyright (C) 2016 Tomonori Kouya                                            */
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
//#ifdef USE_GMP

// (1) L11 * U11 = A11
//
//    start_index
// +--+-----+--------+
// |  +-----+        |
// |  |\ U11|        |
// |  |L11\ |        |
// +  +-----+        |
// |  min_dim        |
// |                 |
// |                 |
// |                 |
// +-----------+-----+
//
int MPFLUdecomp_square(MPFMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t dtmp, dmaxii;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	mpf_init2(dtmp, a->prec);
	mpf_init2(dmaxii, a->prec);

	// dim > min_dim
	for(i = start_index; i < imax; i++)
	{
		mpf_abs(dmaxii, get_mpfmatrix_ij(a, i, i));
		if(mpf_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (MPFLUdecomp_square)!\n", i);
			mpf_clear(dtmp);
			mpf_clear(dmaxii);
			return -1;
		}

		for(j = (i + 1); j < jmax; j++)
		{
			mpf_div(dtmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, i));
			set_mpfmatrix_ij(a, j, i, dtmp);
		}

		for(j = (i + 1); j < jmax; j++)
		{
			for(k = (i + 1); k < jmax; k++)
			{
				mpf_mul(dtmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, k));
				mpf_sub(dtmp, get_mpfmatrix_ij(a, j, k), dtmp);
				set_mpfmatrix_ij(a, j, k, dtmp);
			}
		}
	}

	mpf_clear(dtmp);
	mpf_clear(dmaxii);

	return 0;
}

// (2) Solve L21 * U11 = A21
//
//    start_index
// +--+-----+--------+
// |  +-----+        |
// |  |\ U11|        |
// |  |L11\ |        |
// +  +-----+        |
// |  |     |        |
// |  |     |        |
// |  | L21 |        |
// |  |     |        |
// +--+-----+--------+
//
int MPFLUdecomp_l21(MPFMatrix l21, MPFMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t dtmp, dmaxii;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	mpf_init2(dmaxii, l21->prec);
	mpf_init2(dtmp, l21->prec);

	// dim > min_dim
	for(i = imax; i < dim; i++)
	{
		mpf_abs(dmaxii, get_mpfmatrix_ij(a, i, i));
		if(mpf_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (MPFLUdecomp_l21)!\n", i);
			mpf_clear(dtmp);
			mpf_clear(dmaxii);
			return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			mpf_set(dtmp, get_mpfmatrix_ij(a, i, j));
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_dmatrix_ij(a, i, k) * get_dmatrix_ij(a, k, j);
				mpf_mul(dmaxii, get_mpfmatrix_ij(a, i, k), get_mpfmatrix_ij(a, k, j));
				mpf_sub(dtmp, dtmp, dmaxii);
			}
			mpf_div(dtmp, dtmp, get_mpfmatrix_ij(a, j, j));
			
			//printf("(i - start_index, j - imax) = (%ld, %ld), %ld, %ld\n", i - imax, j - start_index, l21->row_dim, l21->col_dim);
			//printf("(i              , j       ) = (%ld, %ld) %25.17e, %25.17e\n", i, j, dtmp, get_dmatrix_ij(a, j, j));
			set_mpfmatrix_ij(a  , i              , j       , dtmp);
			set_mpfmatrix_ij(l21, i - imax, j - start_index, dtmp);
		}
	}

	mpf_clear(dtmp);
	mpf_clear(dmaxii);

	return 0;
}

// (3) Solve L11 * U12 = A21
//
//    start_index
// +--+-----+--------+
// |  +-----+--------+
// |  |\ U11|  U12   |
// |  |L11\ |        |
// +  +-----+--------+
// |  |     |        |
// |  |     |        |
// |  | L21 |        |
// |  |     |        |
// +--+-----+--------+
//
int MPFLUdecomp_u12(MPFMatrix u12, MPFMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t dtmp, dmaxii;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	mpf_init2(dmaxii, u12->prec);
	mpf_init2(dtmp, u12->prec);

	// dim > min_dim
	for(i = imax; i < dim; i++)
	{
		mpf_abs(dmaxii, get_mpfmatrix_ij(a, i, i));
		if(mpf_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (MPFLUdecomp_u12)!\n", i);
			mpf_clear(dtmp);
			mpf_clear(dmaxii);
			return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			mpf_set(dtmp, get_mpfmatrix_ij(a, j, i));
			//printf("(j              , i              ) = (%ld, %ld) %25.17e\n", j, i , dtmp);
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_dmatrix_ij(a, j, k) * get_dmatrix_ij(a, k, i);
				mpf_mul(dmaxii, get_mpfmatrix_ij(a, j, k), get_mpfmatrix_ij(a, k, i));
				mpf_sub(dtmp, dtmp, dmaxii);
				//printf("(j - start_index, k - start_index) = (%ld, %ld)\n", j - start_index, k - start_index);
			}
			//printf("(j              , i              ) = (%ld, %ld) %25.17e\n", j, i , dtmp);
			set_mpfmatrix_ij(a  , j              , i       , dtmp);
			set_mpfmatrix_ij(u12, j - start_index, i - imax, dtmp);
		}
	}

	mpf_clear(dtmp);
	mpf_clear(dmaxii);

	return 0;
}

// (4) D22 := L21 * U12
// (5) A22 := A22 - D22
//
//    start_index
// +--+-----+--------+
// |  +-----+--------+
// |  |\ U11|  U12   |
// |  |L11\ |        |
// +  +-----+--------+
// |  |     |        |
// |  |     |        |
// |  | L21 |  A22   |
// |  |     |        |
// +--+-----+--------+
//
#define STRASSEN_MIN_DIM 32
int MPFLUdecomp_a22(MPFMatrix a, MPFMatrix d22, MPFMatrix l21, MPFMatrix u12, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];
	MPFVector diag_left, diag_right;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
#if defined(USE_STRASSEN) || defined(USE_WINOGRAD)
#ifdef USE_SCALING
	diag_left = init2_mpfvector(l21->row_dim, d22->prec);
	diag_right = init2_mpfvector(u12->col_dim, d22->prec);

	left_scaling_mpfmatrix(l21, l21, diag_left, NULL);
	right_scaling_mpfmatrix(u12, u12, diag_right, NULL);
#endif

//	mul_mpfmatrix_strassen(d22, l21, u12, min_dim);
//	mul_mpfmatrix_strassen(d22, l21, u12, min_dim / 2);
//	mul_mpfmatrix_strassen(d22, l21, u12, min_dim / 4);
	mul_mpfmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);

#ifdef USE_SCALING
	mul_mpfmatrix_mpfdiag(d22, diag_left, 0, d22, diag_right, 0); 

	mul_mpfmatrix_mpfdiag(l21, diag_left, 1, l21, NULL, 0);
	mul_mpfmatrix_mpfdiag(u12, NULL, 0, u12, diag_right, 1);

	free_mpfvector(diag_left);
	free_mpfvector(diag_right);
#endif

#elif USE_BLOCK
	mul_mpfmatrix_block(d22, l21, u12, STRASSEN_MIN_DIM);
#else
	mul_mpfmatrix_simple(d22, l21, u12);
#endif

	// a22 := a22 - d22
	index[0] = imax;
	index[1] = dim;
	index[2] = jmax;
	index[3] = dim;
	d22_index[0] = 0;
	d22_index[1] = d22->row_dim;
	d22_index[2] = 0;
	d22_index[3] = d22->col_dim;

	sub_mpfmatrix_partial(a, index, a, index, d22, d22_index);

	return 0;
}


/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int MPFLUdecomp_strassen(MPFMatrix a, long int min_dim)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       MPFMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, row_dim, col_dim, dim;
	MPFMatrix l21, u12, d22;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		MPFLUdecomp(a);
		return 0;
	}

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				MPFLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		MPFLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_dmatrix(a);

/*		l21 = init2_mpfmatrix(row_dim, min_dim, a->prec + (a->prec / 4));
		u12 = init2_mpfmatrix(min_dim, col_dim, a->prec + (a->prec / 4));
		d22 = init2_mpfmatrix(row_dim, col_dim, a->prec + (a->prec / 4));
*/		l21 = init2_mpfmatrix(row_dim, min_dim, a->prec);
		u12 = init2_mpfmatrix(min_dim, col_dim, a->prec);
		d22 = init2_mpfmatrix(row_dim, col_dim, a->prec);

		// (2) Solve L21 * U11 = A21
		MPFLUdecomp_l21(l21, a, i, min_dim);
		//print_dmatrix(a);
		//printf("L21:\n");
		//print_dmatrix(l21);

		// (3) Solve L11 * U12 = A12
		MPFLUdecomp_u12(u12, a, i, min_dim);
		//print_dmatrix(a);
		//printf("U12:\n");
		//print_dmatrix(u12);

		//printf("A:\n");
		//print_dmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		MPFLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_dmatrix(d22);
		//print_dmatrix(a);

		// clear
		free_mpfmatrix(l21);
		free_mpfmatrix(u12);
		free_mpfmatrix(d22);

	}

	return 0;
}
