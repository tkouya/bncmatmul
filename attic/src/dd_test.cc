/********************************************************************************/
/* dd_test.cc:Double-double and Quadruple precision Linear Computation Library  */
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
#if 0
#include "mpfr.h"
#include "mpf2mpfr.h"
#endif // 0

#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define QDINLINE
//#include "qd/dd_real.h"
#include "qd/qd_real.h"
#include "qd/fpu.h"


//using namespace std;
using std::cout;
using std::endl;

int main()
{
	int i, j;
	double stime, etime[3];
	dd_real ddval, ddtmp, dd_a, dd_b;
	unsigned int old_cw;
#if 0
	mpf_t mpftmp, mpfval, mpf_a, mpf_b;

	// MPF
	mpf_set_default_prec(128);

	mpf_init(mpfval);
	mpf_init(mpf_a);
	mpf_init(mpf_b);
	//mpf_set_ui(mpf_a, 1UL);
	mpf_sqrt_ui(mpf_a, 2UL); mpf_mul_ui(mpf_a, mpf_a, 2UL);
	mpf_sqrt_ui(mpf_b, 2UL);

	printf("a     = "); mpf_out_str(stdout, 10, 0, mpf_a); printf("\n");
	printf("b     = "); mpf_out_str(stdout, 10, 0, mpf_b); printf("\n");
	printf("a + b = "); mpf_add(mpfval, mpf_a, mpf_b); mpf_out_str(stdout, 10, 0, mpfval); printf("\n");
	printf("a - b = "); mpf_sub(mpfval, mpf_a, mpf_b); mpf_out_str(stdout, 10, 0, mpfval); printf("\n");
	printf("a * b = "); mpf_mul(mpfval, mpf_a, mpf_b); mpf_out_str(stdout, 10, 0, mpfval); printf("\n");
	printf("a / b = "); mpf_div(mpfval, mpf_a, mpf_b); mpf_out_str(stdout, 10, 0, mpfval); printf("\n");

	mpf_clear(mpf_a);
	mpf_clear(mpf_b);
#endif // 0

	// DD
	fpu_fix_start(&old_cw);

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

	fpu_fix_end(&old_cw);

}
