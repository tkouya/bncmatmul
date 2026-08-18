//******************************************************************************
// test_iterative_ref_dd.cpp : Interative refinement method based 
//                             on direct method (dd_real + double)
// Copyright (C) 2019 Tomonori Kouya
// 
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License or any later
// version.
// 
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// 
//******************************************************************************
#include <iostream>
#include <iomanip>

#include <cstdlib>
#include <cmath>

// Multiple precision with QD
#define QD_INLINE
#include "qd/qd_real.h"
#include "qd/fpu.h"

// mplapack, mpblas
//#ifdef USE_MBLAS
#ifdef OLD_MBLAS
    #include <mblas_dd.h>
    #include <mlapack_dd.h>
#else // OLD_MBLAS
    #include <mpblas_dd.h>
    #include <mplapack_dd.h>
#endif // OLD_MBLAS

//#include "bnc.h"
#define USE_DDLINEAR
#include "matmul_strassen.h"

#ifdef USE_OMP
	#include "bncomp.h"
#endif // USE_OMP

// Template linear compucation with double, QD, MPFR/GMP
#include "template_linear.h"

// Time routines
#include "get_secv.h"
/* Simple Estimation of Condition Number */

using namespace std;

int main(int argc, char *argv[])
{
	unsigned long prec;
	int i, j, dimension, *pivot, itimes;
	dd_real *matrix, *true_x, *b, *x;
	dd_real *da_m, *db_m, *dx_m, *dx_true_m;
	dd_real max_relerr, min_relerr, rE2;
    mplapackint *pivot_m, info_m;
	dd_real rtol, atol;
	DDMatrix da;
	DDVector db, dx, dans;
	ddfloat dtmp;

	long int ret_f, ret_d, ret_mpf;
	long int *row_ch, *col_ch;
	double start_time, end_time;

	if(argc <= 1)
	{
		cerr << "USAGE: " << argv[0] << " [dimension]" << endl;
		return EXIT_SUCCESS;
	}

	dimension = atoi(argv[1]);

	if(dimension <= 1)
	{
		cerr << "ERROR: dimension = " << dimension << " is illegal!" << endl;
		return EXIT_FAILURE;
	}

	// initialize
	matrix = new dd_real[dimension * dimension];
	true_x = new dd_real[dimension];
	x      = new dd_real[dimension];
	b      = new dd_real[dimension];
	pivot  = new int[dimension];

	set_test_linear_eq<dd_real>(matrix, true_x, b, dimension);

	// run LU decomposion
	start_time = get_real_secv();
	LU<dd_real>(matrix, dimension, pivot);
	end_time = get_real_secv() - start_time;
	cout << "LU time: " << end_time << ", Dimension: " << dimension << endl;

	// backward & forward substitution
	solve_LU_linear_eq<dd_real>(x, matrix, b, dimension, pivot);

	rE2 = get_minmax_relerr<dd_real>(&max_relerr, &min_relerr, x, true_x, dimension);
	cout << "max, min, norm2_relerr(x) = " << max_relerr << ", " << min_relerr << ", " << rE2 << endl;

	// MPLAPACK
	da_m      = new dd_real[dimension * dimension];
	dx_true_m = new dd_real[dimension];
	dx_m      = new dd_real[dimension];
	db_m      = new dd_real[dimension];
	pivot_m   = new mplapackint[dimension];

	set_test_linear_eq<dd_real>(matrix, dx_true_m, db_m, dimension);
	row2col(da_m, matrix, dimension);

	start_time = get_real_secv();
    Rgetrf(dimension, dimension, da_m, dimension, pivot_m, info_m); //&info_m);
    end_time = get_real_secv();
    Rgetrs("N", dimension, 1, da_m, dimension, pivot_m, db_m, dimension, info_m); //&info_m);
    printf("Rgetrf time : %f, Dimension: %d\n", end_time - start_time, dimension);
	//cout << "rE2(x) = " << get_relerr_norm2<dd_real>(db_m, dx_true_m, dimension) << endl;
	rE2 = get_minmax_relerr<dd_real>(&max_relerr, &min_relerr, db_m, dx_true_m, dimension);
	cout << "max, min, norm2_relerr(x) = " << max_relerr << ", " << min_relerr << ", " << rE2 << endl;

	// free
	delete_array<dd_real>(da_m, dimension * dimension);
	delete_array<dd_real>(dx_true_m, dimension);
	delete_array<dd_real>(dx_m, dimension);
	delete_array<dd_real>(db_m, dimension);
	delete pivot_m;

	// BNCmatmul
	// initialize
	row_ch = (long int *)calloc(sizeof(long int), dimension);
	da = init_ddmatrix(dimension, dimension);
	db = init_ddvector(dimension);
	dx = init_ddvector(dimension);

	set_test_linear_eq<dd_real>(matrix, true_x, b, dimension);
	for(i = 0; i < dimension; i++)
	{
		for(j = 0; j < dimension; j++)
		{
			dtmp.val[0] = matrix[ROW_ZERO_IJ(i, j, dimension, dimension)].x[0];
			dtmp.val[1] = matrix[ROW_ZERO_IJ(i, j, dimension, dimension)].x[1];
			
			set_ddmatrix_ij(da, i, j, dtmp.val);
		}
		dtmp.val[0] = b[i].x[0];
		dtmp.val[1] = b[i].x[1];

		set_ddvector_i(db, i, dtmp.val);
	}

	start_time = get_real_secv();
	ret_d = DDLUdecompPM(da, row_ch);
	end_time = get_real_secv() - start_time;
	ret_d = SolveDDLSPM(dx, da, db, row_ch);

	for(i = 0; i < dimension; i++)
	{
		dtmp = get_ddvector_i_ddfloat(dx, i);

		x[i].x[0] = dtmp.val[0];
		x[i].x[1] = dtmp.val[1];
	}

	printf("DDLUdecompPM: %f, Dim = %d\n", end_time, dimension);
	rE2 = get_minmax_relerr<dd_real>(&max_relerr, &min_relerr, x, true_x, dimension);
	cout << "max, min, norm2_relerr(x) = " << max_relerr << ", " << min_relerr << ", " << rE2 << endl;

	return 0;

	// double-DD
	// set test problem
	set_test_linear_eq<dd_real>(matrix, true_x, b, dimension);
	rtol = "1.0e-50"; atol = "0.0";

	itimes = iterative_refinement<dd_real, double>(x, matrix, b, rtol, atol, dimension, dimension * 10);
	cout << "Iterative Times: " << itimes << ", Dimension: " << dimension << endl;

	// print solution
	for(i = 0; i < dimension; i++)
		cout << setw(3) << i << " " << scientific << setprecision(32) << x[i] << " " << setprecision(3) << get_relerr(x[i], true_x[i]) << endl;

	// free
	delete_array<dd_real>(matrix, dimension * dimension);
	delete_array<dd_real>(true_x, dimension);
	delete_array<dd_real>(x, dimension);
	delete_array<dd_real>(b, dimension);
	delete pivot;

	return EXIT_SUCCESS;
}
