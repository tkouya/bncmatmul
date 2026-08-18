/********************************************************************************/
/* gqd_test.cu:Double-double and Quadruple precision Linear Computation Library */
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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>

// gdd, gqd class
#include "gqd.cu" 
#include "cuda_header.cu"
#include "gqd_type.h" // G[DQ]DSTART, END

void gdd_test(void)
{
	gdd_real gddval, gddtmp, gdd_a, gdd_b;

	// Initialize
	GDDStart();

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

	// Finalize
	GDDEnd();
}

void gqd_test(void)
{
	gqd_real qdval, qdtmp, qd_a, qd_b;

	// Initialize
	GQDStart();

	// Finalize
	GQDEnd();
}
