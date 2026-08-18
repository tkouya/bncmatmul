/********************************************************************************/
/* gdd_test.cc:Double-double and Quadruple precision Linear Computation Library */
/* Copyright (C) 2015 Tomonori Kouya                                            */
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
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define QDINLINE

// CPU
#include "qd/qd_real.h"
#include "qd/fpu.h"

// GPU
//#include "

//using namespace std;
using std::cout;
using std::endl;

int main()
{
	int i, j;
	double stime, etime[3];
	dd_real ddval, ddtmp, dd_a, dd_b;
	qd_real qdval, qdtmp, qd_a, qd_b;
	unsigned int old_cw;

	// DD & QD on CPU
	fpu_fix_start(&old_cw);

	// DD
//	dd_a = static_cast<double>(1.0);
	//dd_a = dd_real::sqrt((int)2) * 2;
	dd_a = sqrt((int)2) * 2;
//	dd_b = dd_real(2.0);
	dd_b = dd_real::sqrt((int)2);

	std::cout.precision(dd_real::_ndigits);
	std::cout << "a     = " << dd_a << endl;
	std::cout << "b     = " << dd_b << "\n";

	ddval = (dd_real)dd_a;
	ddval += (dd_real)dd_b;
	std::cout << "a + b = " << dd_real(dd_a) + dd_real(dd_b) << "\n";
	std::cout << "a + b = " << ddval << "\n";

	std::cout << "a - b = " << dd_a - dd_b << "\n";
	std::cout << "a * b = " << (const dd_real &)dd_a * (const dd_real &)dd_b << "\n";
	std::cout << "a * b = " << (dd_real)operator*((const dd_real &)dd_a, (const dd_real &)dd_b) << "\n";
	std::cout << "a / b = " << dd_a / dd_b << "\n";

	// QD
//	qd_a = static_cast<double>(1.0);
	//qd_a = qd_real::sqrt((int)2) * 2;
	qd_a = sqrt((int)2) * 2;
//	qd_b = qd_real(2.0);
	qd_b = sqrt((int)2);

	std::cout.precision(qd_real::_ndigits);
	std::cout << "a     = " << qd_a << endl;
	std::cout << "b     = " << qd_b << "\n";

	qdval = (qd_real)dd_a;
	qdval += (qd_real)dd_b;
	std::cout << "a + b = " << qd_real(qd_a) + qd_real(qd_b) << "\n";
	std::cout << "a + b = " << qdval << "\n";

	std::cout << "a - b = " << qd_a - qd_b << "\n";
	std::cout << "a * b = " << (const qd_real &)qd_a * (const qd_real &)qd_b << "\n";
	std::cout << "a * b = " << (qd_real)operator*((const qd_real &)qd_a, (const qd_real &)qd_b) << "\n";
	std::cout << "a / b = " << qd_a / qd_b << "\n";

	fpu_fix_end(&old_cw);

}
