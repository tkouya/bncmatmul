//******************************************************************************
// test_iterative_ref_mpreal.cpp : Interative refinement method based 
//                         on direct method (double, dd_real and qd_real + MPFR)
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

// mplapack
//#ifdef USE_MBLAS
    #include <mpblas_mpfr.h>
    #include <mplapack_mpfr.h>
//#endif // USE_MBLAS
// mpreal
#include "mpreal.h"
//#define __MPREAL_H__
// Template linear compucation with double, QD, MPFR/GMP
#define _QD_QD_REAL_H // set_array(dd, mpreal) etc.
#define __MPREAL_H__
#include "template_linear.h"

// Time routines
#include "get_secv.h"
/* Simple Estimation of Condition Number */

using namespace std;
using namespace mpfr;

int main(int argc, char *argv[])
{
	unsigned long prec;
	int i, j, dimension, *pivot, itimes;
	mpreal *matrix, *true_x, *b, *x;
	mpreal rtol, atol;
	dd_real *ddmatrix, *ddtrue_x, *ddb, *ddx;
	qd_real *qdmatrix, *qdtrue_x, *qdb, *qdx;

	double start_time, end_time;
	unsigned int old_cw;
	char fname_A[256], fname_true_x[256], fname_vec_b[256];
	mpreal *da_m;
    mplapackint *pivot_m, info;

	if(argc <= 2)
	{
		cerr << "USAGE: " << argv[0] << " [dimension] [prec]" << endl;
		return EXIT_SUCCESS;
	}

	prec = atoi(argv[2]);
	if(prec <= 1)
	{
		cerr << "ERROR: prec = " << prec << " is illegal!" << endl;
		return EXIT_FAILURE;
	}
	//mpreal::set_default_prec(prec);
	mpfr_set_default_prec((mp_prec_t)prec);
    mpfr::mpreal::default_prec = (mp_prec_t)prec;

	dimension = atoi(argv[1]);

	if(dimension <= 1)
	{
		cerr << "ERROR: dimension = " << dimension << " is illegal!" << endl;
		return EXIT_FAILURE;
	}

	fpu_fix_start(&old_cw);

	// initialize
	matrix = new mpreal[dimension * dimension];
	true_x = new mpreal[dimension];
	x      = new mpreal[dimension];
	b      = new mpreal[dimension];
	pivot  = new int[dimension];

	//set_test_linear_eq<mpreal>(matrix, true_x, b, dimension);

	sprintf(fname_A, "../bncmatmul-0.21/python/mat_a_%d_%d.txt", dimension, dimension);
	sprintf(fname_true_x, "../bncmatmul-0.21/python/vec_true_x_%d.txt", dimension);
	sprintf(fname_vec_b, "../bncmatmul-0.21/python/vec_b_%d.txt", dimension);	

	read_test_linear_eq(matrix, true_x, b, dimension, fname_A, fname_true_x, fname_vec_b);

	// dd run LU decomposion
	// initialize
	ddmatrix = new dd_real[dimension * dimension];
	ddtrue_x = new dd_real[dimension];
	ddx      = new dd_real[dimension];
	ddb      = new dd_real[dimension];
	set_array(ddmatrix, matrix, dimension * dimension);
	set_array(ddtrue_x, true_x, dimension);
	set_array(ddx, x, dimension);
	set_array(ddb, b, dimension);

	start_time = get_secv();
	LU<dd_real>(ddmatrix, dimension, pivot);

	// backward & forward substitution
	solve_LU_linear_eq<dd_real>(ddx, ddmatrix, ddb, dimension, pivot);
	end_time = get_secv() - start_time;

	cout << "dd comp.time(second): " << end_time << ", relerr = " << get_relerr_norm2(ddx, ddtrue_x, dimension) << endl;

	// qd run LU decomposion
	// initialize
	qdmatrix = new qd_real[dimension * dimension];
	qdtrue_x = new qd_real[dimension];
	qdx      = new qd_real[dimension];
	qdb      = new qd_real[dimension];
	set_array(qdmatrix, matrix, dimension * dimension);
	set_array(qdtrue_x, true_x, dimension);
	set_array(qdx, x, dimension);
	set_array(qdb, b, dimension);

	start_time = get_secv();
	LU<qd_real>(qdmatrix, dimension, pivot);

	// backward & forward substitution
	solve_LU_linear_eq<qd_real>(qdx, qdmatrix, qdb, dimension, pivot);
	end_time = get_secv() - start_time;

	cout << "qd comp.time(second): " << end_time << ", relerr = " << get_relerr_norm2(qdx, qdtrue_x, dimension) << endl;


	// mpreal run LU decomposion
	start_time = get_secv();
	LU<mpreal>(matrix, dimension, pivot);

	// backward & forward substitution
	solve_LU_linear_eq<mpreal>(x, matrix, b, dimension, pivot);
	end_time = get_secv() - start_time;

	cout << "mpreal comp.time(second): " << end_time << ", relerr = " << get_relerr_norm2(x, true_x, dimension) << endl;

    // MPLAPACK
	da_m = new mpreal[dimension * dimension];
	//pivot_m = new mplapackint[dimension];

	read_test_linear_eq(matrix, true_x, b, dimension, fname_A, fname_true_x, fname_vec_b);
	row2col(da_m, matrix, dimension);

    start_time = get_secv();
    Rgetrf((mplapackint)dimension, (mplapackint)dimension, da_m, (mplapackint)dimension, pivot_m, info); //&info);
    Rgetrs("N", (mplapackint)dimension, 1, da_m, (mplapackint)dimension, pivot_m, b, (mplapackint)dimension, info); //&info);
    end_time = get_secv() - start_time;
	cout << "MPLAPACK comp.time(second): " << end_time << ", relerr = " << get_relerr_norm2(b, true_x, dimension) << endl;
 
	// double-MPFR
	// set test problem
	//set_test_linear_eq<mpreal>(matrix, true_x, b, dimension);
	read_test_linear_eq(matrix, true_x, b, dimension, fname_A, fname_true_x, fname_vec_b);//	rtol = "1.0e-300"; atol = "0.0";
	rtol = "1.0e-108"; atol = "0.0";
	cout << "rtol, atol = " << rtol <<", " << atol << endl;
//	rtol = "1.0e-50"; atol = "0.0";

/*	start_time = get_secv();
	itimes = iterative_refinement<mpfr::mpreal, double>(x, matrix, b, rtol, atol, dimension, dimension * 10);
	end_time = get_secv() - start_time;
	cout << "Iterative Times: " << itimes << ", Dimension: " << dimension << endl;
	cout << "comp.time(second): " << end_time << ", relerr = " << get_relerr_norm2(x, true_x, dimension) << endl;
*/
	// print solution
//	for(i = 0; i < dimension; i++)
//		cout << setw(3) << i << " " << scientific << setprecision(mpfr::bits2digits(mpreal::get_default_prec())) << x[i] << " " << setprecision(3) << get_relerr(x[i], true_x[i]) << endl;

	// DD-MPFR
	// set test problem
	//set_test_linear_eq<mpreal>(matrix, true_x, b, dimension);
	read_test_linear_eq(matrix, true_x, b, dimension, fname_A, fname_true_x, fname_vec_b);//	rtol = "1.0e-100"; atol = "0.0";
	start_time = get_secv();
	itimes = iterative_refinement<mpfr::mpreal, dd_real>(x, matrix, b, rtol, atol, dimension, dimension);
	end_time = get_secv() - start_time;
	cout << "Iterative Times: " << itimes << ", Dimension: " << dimension << endl;
	cout << "comp.time(second): " << end_time << ", relerr = " << get_relerr_norm2(x, true_x, dimension) << endl;

	// print solution
//	for(i = 0; i < dimension; i++)
//		cout << setw(3) << i << " " << scientific << setprecision(mpfr::bits2digits(mpreal::get_default_prec())) << x[i] << " " << setprecision(3) << get_relerr(x[i], true_x[i]) << endl;

	// QD-MPFR
	// set test problem
	//set_test_linear_eq<mpreal>(matrix, true_x, b, dimension);
	read_test_linear_eq(matrix, true_x, b, dimension, fname_A, fname_true_x, fname_vec_b);//	rtol = "1.0e-100"; atol = "0.0";
	start_time = get_secv();
//	rtol = "1.0e-200"; atol = "0.0";
	itimes = iterative_refinement<mpfr::mpreal, qd_real>(x, matrix, b, rtol, atol, dimension, dimension);
	end_time = get_secv() - start_time;
	cout << "Iterative Times: " << itimes << ", Dimension: " << dimension << endl;
	cout << "comp.time(second): " << end_time << ", relerr = " << get_relerr_norm2(x, true_x, dimension) << endl;

	// print solution
//	for(i = 0; i < dimension; i++)
//		cout << setw(3) << i << " " << scientific << setprecision(mpfr::bits2digits(mpreal::get_default_prec())) << x[i] << " " << setprecision(3) << get_relerr(x[i], true_x[i]) << endl;

	fpu_fix_end(&old_cw);

	// free
	delete_array<mpreal>(matrix, dimension * dimension);
	delete_array<mpreal>(true_x, dimension);
	delete_array<mpreal>(x, dimension);
	delete_array<mpreal>(b, dimension);
	delete pivot;

	return EXIT_SUCCESS;
}
