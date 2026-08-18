/********************************************************************************/
/*                                                                              */
/* example_readmatrix.c :                                                       */
/* Copyright (c) 2006-2011 Tomonori Kouya, All rights reserved.                 */
/*                                                                              */
/* 2011-08-26 Version 0.0: separate example_readmatrix.c from readmatrix.c      */
/* 2011-08-29 Version 0.1: use functions defined in writematrix.c               */
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

#include "bncmm.h"

int main(int argc, char *argv[])
{
	DMatrix dmat;
	DRSMatrix dsmat;
#ifdef USE_GMP
	MPFMatrix mpfmat;
	MPFRSMatrix mpfsmat;
#endif

	// Dense Matrix
    if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}

	dmat = init_dmatrix_readMMcoordinate(argv[1]);
	dsmat = init_set_drsmatrix_dmatrix(dmat);

	if(dmat != NULL)
	{
//		print_dmatrix(dmat);
		print_drsmatrix(dsmat);

		writeMMcoordinate_dmatrix("dmat.mtx", dmat);

		free_dmatrix(dmat);
		free_drsmatrix(dsmat);
	}

	// Sparse Matrix
    if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}

	dsmat = init_drsmatrix_readMMcoordinate(argv[1]);

	if(dsmat != NULL)
	{
		print_drsmatrix(dsmat);

		// writematrix
		writeMMcoordinate_drsmatrix("dsmat.mtx", dsmat);

		free_drsmatrix(dsmat);
	}


#ifdef USE_GMP

	set_bnc_default_prec(128);

	// Dense Matrix
    if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}

	mpfmat = init_mpfmatrix_readMMcoordinate(argv[1]);
	mpfsmat = init_set_mpfrsmatrix_mpfmatrix(mpfmat);

	if(mpfmat != NULL)
	{
//		print_mpfmatrix(mpfmat);
		print_mpfrsmatrix(mpfsmat);

		writeMMcoordinate_mpfmatrix("mpfmat.mtx", mpfmat);

		free_mpfmatrix(mpfmat);
		free_mpfrsmatrix(mpfsmat);
	}

	// Sparse Matrix
    if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}

	mpfsmat = init_mpfrsmatrix_readMMcoordinate(argv[1]);

	if(mpfsmat != NULL)
	{
		print_mpfrsmatrix(mpfsmat);

		writeMMcoordinate_mpfrsmatrix("mpfsmat.mtx", mpfsmat);

		free_mpfrsmatrix(mpfsmat);
	}
#endif

	return 0;
}
