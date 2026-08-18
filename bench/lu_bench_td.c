/********************************************************************************/
/* lu_bench_td.c:                                                               */
/* Copyright (C) 2022-2026 Tomonori Kouya                                       */
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
#ifndef _STDIO_H
#include <stdio.h>
#endif

#ifndef _MATH_H
#include <math.h>
#endif

#include "bncomp.h"
#include "matmul_strassen.h"
#include "get_secv.h"

// b := A * ans, with A a Frank (or Lotkin) matrix and ans = [0, 1, ..., dim-1]
void get_tdproblem(TDMatrix a, TDVector b, TDVector ans)
{
	long int i;

#ifdef USE_LOTKIN
	lotkin_tdmatrix(a, a->row_dim);
#else
	frank_tdmatrix(a, a->row_dim);
#endif // USE_LOTKIN

	for(i = 0; i < ans->dim; i++)
		set_tdvector_i_d(ans, i, (double)i);

	mul_tdmatrix_tdvec(b, a, ans);
}

int main(int argc, char *argv[])
{
	long int dim, *pivot;
	int num_threads;
	double stime, etime[6];
	TDMatrix tda;
	TDVector tdb, tdx, tdans;
	double relerr[3][TDSIZE];
	long int ret_td;
	long int block_size, min_dim, limit_dim;
#ifdef USE_OZ
	#ifndef OZ_MAX_NUM_DIV
	#define OZ_MAX_NUM_DIV 4
	#endif
	int max_num_div = OZ_MAX_NUM_DIV;
#endif

	if(argc <= 1)
	{
#ifndef _OPENMP
		fprintf(stderr, "Usage: %s [dim]\n", argv[0]);
#else // _OPENMP
		fprintf(stderr, "Usage: %s [dim] [#threads]\n", argv[0]);
#endif // _OPENMP
		return 0;
	}
	dim = atol(argv[1]);
	if(dim <= 0)
		return 0;

#ifdef _OPENMP
	num_threads = 1;
	if(argc >= 3)
	{
		num_threads = atoi(argv[2]);
		if(num_threads < 1)
			num_threads = 1;
	}
	set_bncomp_num_threads(num_threads);
#endif // _OPENMP

#ifdef USE_OZ
	printf("Ozaki scheme, max_num_div = %d\n", max_num_div);
#endif // USE_OZ

	fpu_fix_start(NULL);

	/* initialize */
	tda = init_tdmatrix(dim, dim);
	tdb = init_tdvector(dim);
	tdx = init_tdvector(dim);
	tdans = init_tdvector(dim);
	pivot = (long int *)calloc(dim, sizeof(long int));

	/* normal LU */
	get_tdproblem(tda, tdb, tdans);
	stime = get_real_secv();
#ifndef _OPENMP
	ret_td = TDLUdecompPM(tda, pivot);
#else // _OPENMP
	ret_td = TDLUdecompPM_omp(tda, pivot);
#endif // _OPENMP
	etime[4] = get_real_secv() - stime;
	ret_td = SolveTDLSPM(tdx, tda, tdb, pivot);
	printf("normalLU(%5ld): %10.3f, ", dim, etime[4]);
	relerr_element_tdvector(relerr[0], relerr[1], relerr[2], tdx, tdans, 0);
	printf("%10.3e, %10.3e, %10.3e\n", relerr[0][0], relerr[1][0], relerr[2][0]);

	min_dim = STRASSEN_MIN_DIM;
	etime[5] = 0.0;
	limit_dim = dim;

	for(block_size = min_dim; block_size < limit_dim; block_size += min_dim)
	{
		get_tdproblem(tda, tdb, tdans);
		stime = get_real_secv();
#ifndef _OPENMP
		#ifdef USE_OZ
		ret_td = TDLUdecomp_ozPM(tda, pivot, block_size, max_num_div);
		#else // USE_OZ
		ret_td = TDLUdecomp_strassenPM(tda, pivot, block_size);
		#endif // USE_OZ
#else // _OPENMP
		#ifdef USE_OZ
		ret_td = TDLUdecomp_ozPM_omp(tda, pivot, block_size, max_num_div);
		#else // USE_OZ
		ret_td = TDLUdecomp_strassenPM_omp(tda, pivot, block_size);
		#endif // USE_OZ
#endif // _OPENMP
		etime[5] = get_real_secv() - stime;
		ret_td = SolveTDLSPM(tdx, tda, tdb, pivot);
#ifdef USE_BLOCK
		printf("blockLU   (%5ld, %5ld): %10.3f, ", dim, block_size, etime[5]);
#elif defined(USE_OZ)
		printf("ozakiLU   (%5ld, %5ld): %10.3f, ", dim, block_size, etime[5]);
#else
		printf("strassenLU(%5ld, %5ld): %10.3f, ", dim, block_size, etime[5]);
#endif
		relerr_element_tdvector(relerr[0], relerr[1], relerr[2], tdx, tdans, 0);
		printf("%10.3e, %10.3e, %10.3e\n", relerr[0][0], relerr[1][0], relerr[2][0]);
	}

	free_tdmatrix(tda);
	free_tdvector(tdb);
	free_tdvector(tdx);
	free_tdvector(tdans);
	free(pivot);

	return 0;
}
