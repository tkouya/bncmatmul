/********************************************************************************/
/* bncomp_linear.c: Parallelized Mutiple Precision Linear Computation Library   */
/*                                                                  with OpenMP */
/* Copyright (C) 2013-2020 Tomonori Kouya                                       */
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
// BNCpack with OpenMP
#include "bncomp.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// set_bncomp_num_threads(int num_threads)
int set_bncomp_num_threads(int num_threads)
{	
#ifdef _OPENMP
	if((num_threads > 0) && (num_threads <= BNCOMP_MAX_NUM_THREADS))
	{
		_bncomp_num_threads = num_threads;
		omp_set_num_threads(_bncomp_num_threads);
	}
	/* NOTE: report omp_get_max_threads(), NOT omp_get_num_threads(): the latter
	 * is 1 outside any parallel region (which is where this function is normally
	 * called from), so it printed a misleading "1 threads will be used". */
	printf("Max.Num.Threads           : %d\n", omp_get_max_threads());
	printf("OMP %d threads will be used.\n", omp_get_max_threads());
#else // _OPENMP
	_bncomp_num_threads = 1;
	printf("Max.Num.Threads           : %d\n", 1);
	printf("OMP %d threads will be used.\n", 1);
#endif // _OPENMP

	return _bncomp_num_threads;
}

// get number of threads in BNCOMP
int get_bncomp_num_threads(void)
{
	return _bncomp_num_threads;
}

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus