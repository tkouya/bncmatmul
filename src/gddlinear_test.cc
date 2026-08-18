/********************************************************************************/
/* gddlinear_test.cc: Double-double and Quadruple precision                     */
/*                                 Linear Computation Library with GQD and CUDA */
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
#include "gddlinear.h"

using namespace std;

//#define ROW_DIM 10
//#define ROW_DIM 32
//#define ROW_DIM 128
//#define ROW_DIM 512
#define ROW_DIM 1024
#define COL_DIM ROW_DIM

int main()
{
	int i, j;
	double stime, etime[4];
	float cuda_etime[4];
	cudaEvent_t start, stop;
	DDVector ddvec_a, ddvec_b, ddvec_c;
	DDMatrix ddmat_a, ddmat_b, ddmat_c;
	dd_real ddval;
	QDVector qdvec_a, qdvec_b, qdvec_c;
	QDMatrix qdmat_a, qdmat_b, qdmat_c;
	qd_real qdval;
	int num_blocks, num_threads;
	GDDVector gddvec_a, gddvec_b, gddvec_c;
	gdd_real gddval, *ptr_gddval_dev;
	GDDMatrix gddmat_a, gddmat_b, gddmat_c;
	GQDVector gqdvec_a, gqdvec_b, gqdvec_c;
	gqd_real gqdval, *ptr_gqdval_dev;
	GQDMatrix gqdmat_a, gqdmat_b, gqdmat_c;

	// Initialize QD library
	fpu_fix_start(NULL);

	// DD & GDD
	ddvec_a = init_ddvector(ROW_DIM);
	ddvec_b = init_ddvector(ROW_DIM);
	ddvec_c = init_ddvector(ROW_DIM);

	for(i = 0; i < ROW_DIM; i++)
	{
		// val := sqrt((i + 1));
		ddvec_a->element[i] = dd_real::sqrt((int)(i + 1));
		ddvec_b->element[i] = dd_real::sqrt((int)(i + 1));
		//ddvec->element[i] = sqrt(qdval);
	}

	print_ddvector(ddvec_a);
	norm2_ddvector(&ddval, ddvec_a);
	printf("DD : ||vec||_2 = ");
	cout.precision(dd_real::_ndigits);
	cout << ddval << "\n";

	// GDD start!
	GDDStart();

//	num_blocks = 1;  // #blocks per grid
//	num_blocks = 2;  // #blocks per grid
//	num_blocks = 4;  // #blocks per grid
	num_blocks = 8;  // #blocks per grid
//	num_threads = 16; // #threads per block
	num_threads = 64; // #threads per block

	gddvec_a = init_gddvector_dev(ROW_DIM);
	gddvec_b = init_gddvector_dev(ROW_DIM);
	gddvec_c = init_gddvector_dev(ROW_DIM);
	cudaMalloc((void **)&ptr_gddval_dev, sizeof(gdd_real));

	// gddvec := ddvec
	subst_gddvector_dev_ddvec(gddvec_a, ddvec_a);
	subst_gddvector_dev_ddvec(gddvec_b, ddvec_b);
	//subst_ddvector_gddvec_dev(ddvec_c, gddvec_c);

	// ----------
	// c := a + b
	// ----------
	// DD
	printf(" DD: c := a + b\n");
	add_ddvector(ddvec_c, ddvec_a, ddvec_b);
	print_ddvector(ddvec_c);

	// GDD
	printf("GDD: c := a + b\n");
	add_gddvector_dev(gddvec_c, gddvec_a, gddvec_b, num_blocks, num_threads);
	print_gddvector_dev(gddvec_c);

	// ----------
	// c := a - b
	// ----------
	// DD
	printf(" DD: c := a - b\n");
	sub_ddvector(ddvec_c, ddvec_a, ddvec_b);
	print_ddvector(ddvec_c);

	// GDD
	printf("GDD: c := a - b\n");
	sub_gddvector_dev(gddvec_c, gddvec_a, gddvec_b, num_blocks, num_threads);
	print_gddvector_dev(gddvec_c);

	// ----------
	// c := val * a
	// ----------
	// DD
	ddval = (dd_real)2.0;
	printf(" DD: c := val * a\n");
	cmul_ddvector(ddvec_c, ddval, ddvec_a);
	print_ddvector(ddvec_c);

	// GDD
	dd2gdd(&gddval, &ddval);
	printf("GDD: c := val * a\n");
	cmul_gddvector_dev(gddvec_c, gddval, gddvec_a, num_blocks, num_threads);
	print_gddvector_dev(gddvec_c);

	// ----------
	// (a, b)
	// ----------
	// DD
	ip_ddvector(&ddval, ddvec_a, ddvec_b);
	printf(" DD : (a, b) = ");
	cout.precision(dd_real::_ndigits);
	cout << ddval << "\n";

	// GDD
	ip_gddvector_dev(ptr_gddval_dev, gddvec_a, gddvec_b, num_blocks, num_threads);
	//ip_gddvector_dev(ptr_gddval_dev, gddvec_a, gddvec_b, 1, num_threads);
	gdd2dd_dev(&ddval, ptr_gddval_dev);
	printf("GDD : (a, b) = ");
	cout.precision(dd_real::_ndigits);
	cout << ddval << "\n";

	// ----------
	// ||a||_2
	// ----------
	// DD
	norm2_ddvector(&ddval, ddvec_a);
	printf(" DD : ||a||_2 = ");
	cout.precision(dd_real::_ndigits);
	cout << ddval << "\n";

	// GDD
	norm2_gddvector_dev(ptr_gddval_dev, gddvec_a, num_blocks, num_threads);
	//norm2_gddvector_dev(ptr_gddval_dev, gddvec_a, 1, num_threads);
	gdd2dd_dev(&ddval, ptr_gddval_dev);
	printf("GDD : ||a||_2 = ");
	cout.precision(dd_real::_ndigits);
	cout << ddval << "\n";

/* Matrix */
	// DD & GDD
	ddmat_a = init_ddmatrix(ROW_DIM, COL_DIM);
	ddmat_b = init_ddmatrix(ROW_DIM, COL_DIM);
	ddmat_c = init_ddmatrix(ROW_DIM, COL_DIM);

	gddmat_a = init_gddmatrix_dev(ROW_DIM, COL_DIM);
	gddmat_b = init_gddmatrix_dev(ROW_DIM, COL_DIM);
	gddmat_c = init_gddmatrix_dev(ROW_DIM, COL_DIM);

	for(i = 0; i < ROW_DIM; i++)
	{
		for(j = 0; j < COL_DIM; j++)
		{
			// val := sqrt((i + 1));
			ddmat_a->element[i * COL_DIM + j] = dd_real::sqrt((int)(i + j + 1));
			ddmat_b->element[i * COL_DIM + j] = dd_real::sqrt((int)(i + j + 1));
		}
	}
	subst_gddmatrix_dev_ddmat(gddmat_a, ddmat_a);
	subst_gddmatrix_dev_ddmat(gddmat_b, ddmat_b);

	// Print gddmatrix
	printf("ddmat_a:\n");
	normf_ddmatrix(&ddval, ddmat_a);
	cout.precision(dd_real::_ndigits);
	cout << ddval << "\n";

	printf("gddmat_a:\n");
	//print_gddmatrix_dev(gddmat_a);
	normf_gddmatrix_dev(ptr_gddval_dev, gddmat_a, 1, num_threads);
	gdd2dd_dev(&ddval, ptr_gddval_dev);
	cout.precision(dd_real::_ndigits);
	cout << ddval << "\n";

	printf("ddmat_b:\n");
	normf_ddmatrix(&ddval, ddmat_b);
	cout.precision(dd_real::_ndigits);
	cout << ddval << "\n";

	printf("gddmat_b:\n");
	//print_gddmatrix_dev(gddmat_b);
	normf_gddmatrix_dev(ptr_gddval_dev, gddmat_b, num_blocks, num_threads);
	gdd2dd_dev(&ddval, ptr_gddval_dev);
	cout.precision(dd_real::_ndigits);
	cout << ddval << "\n";

	// ----------
	// C := A * B
	// ----------
	// DD
	stime = get_real_secv();
	mul_ddmatrix(ddmat_c, ddmat_a, ddmat_b);
	etime[0] = get_real_secv() - stime;

	printf("DD : || A * B ||_F: %10.3g\n", etime[0]);
	normf_ddmatrix(&ddval, ddmat_c);
	cout.precision(dd_real::_ndigits);
	cout << ddval << "\n";

	// GDD
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	stime = get_real_secv();
	mul_gddmatrix_dev(gddmat_c, gddmat_a, gddmat_b, num_blocks, num_threads);
	etime[1] = get_real_secv() - stime;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cuda_etime[1], start, stop); // Unit: ms
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("GDD: || A * B ||_F: %10.3g, %10.3g\n", etime[1], cuda_etime[1] / 1000.0);
	normf_gddmatrix_dev(ptr_gddval_dev, gddmat_c, num_blocks, num_threads);
	gdd2dd_dev(&ddval, ptr_gddval_dev);
	cout.precision(dd_real::_ndigits);
	cout << ddval << "\n";

/* Free! */
	cudaFree(ptr_gddval_dev);
	free_gddvector_dev(gddvec_a);
	free_gddvector_dev(gddvec_b);
	free_gddvector_dev(gddvec_c);

	free_gddmatrix_dev(gddmat_a);
	free_gddmatrix_dev(gddmat_b);
	free_gddmatrix_dev(gddmat_c);

	// GDD end!
	GDDEnd();

	// Free DDVectors
	free_ddvector(ddvec_a);
	free_ddvector(ddvec_b);
	free_ddvector(ddvec_c);

	free_ddmatrix(ddmat_a);
	free_ddmatrix(ddmat_b);
	free_ddmatrix(ddmat_c);

	// Initialize QD library
	//fpu_fix_start(NULL);

	// DD & GQD
	qdvec_a = init_qdvector(ROW_DIM);
	qdvec_b = init_qdvector(ROW_DIM);
	qdvec_c = init_qdvector(ROW_DIM);

	for(i = 0; i < ROW_DIM; i++)
	{
		// val := sqrt((i + 1));
		qdvec_a->element[i] = (qd_real)sqrt((qd_real)(i + 1));
		qdvec_b->element[i] = (qd_real)sqrt((qd_real)(i + 1));
		//qdvec->element[i] = sqrt(qdval);
	}

	print_qdvector(qdvec_a);
	norm2_qdvector(&qdval, qdvec_a);
	printf("DD : ||vec||_2 = ");
	cout.precision(qd_real::_ndigits);
	cout << qdval << "\n";

	// GQD start!
	GQDStart();

//	num_blocks = 1;  // #blocks per grid
//	num_blocks = 2;  // #blocks per grid
	num_blocks = 4;  // #blocks per grid
//	num_threads = 16; // #threads per block
	num_threads = 64; // #threads per block

	gqdvec_a = init_gqdvector_dev(ROW_DIM);
	gqdvec_b = init_gqdvector_dev(ROW_DIM);
	gqdvec_c = init_gqdvector_dev(ROW_DIM);
	cudaMalloc((void **)&ptr_gqdval_dev, sizeof(gqd_real));

	// gqdvec := qdvec
	subst_gqdvector_dev_qdvec(gqdvec_a, qdvec_a);
	subst_gqdvector_dev_qdvec(gqdvec_b, qdvec_b);
	//subst_qdvector_gqdvec_dev(qdvec_c, gqdvec_c);

	// ----------
	// c := a + b
	// ----------
	// DD
	printf(" DD: c := a + b\n");
	add_qdvector(qdvec_c, qdvec_a, qdvec_b);
	print_qdvector(qdvec_c);

	// GQD
	printf("GQD: c := a + b\n");
	add_gqdvector_dev(gqdvec_c, gqdvec_a, gqdvec_b, num_blocks, num_threads);
	print_gqdvector_dev(gqdvec_c);

	// ----------
	// c := a - b
	// ----------
	// DD
	printf(" DD: c := a - b\n");
	sub_qdvector(qdvec_c, qdvec_a, qdvec_b);
	print_qdvector(qdvec_c);

	// GQD
	printf("GQD: c := a - b\n");
	sub_gqdvector_dev(gqdvec_c, gqdvec_a, gqdvec_b, num_blocks, num_threads);
	print_gqdvector_dev(gqdvec_c);

	// ----------
	// c := val * a
	// ----------
	// DD
	qdval = (qd_real)2.0;
	printf(" DD: c := val * a\n");
	cmul_qdvector(qdvec_c, qdval, qdvec_a);
	print_qdvector(qdvec_c);

	// GQD
	qd2gqd(&gqdval, &qdval);
	printf("GQD: c := val * a\n");
	cmul_gqdvector_dev(gqdvec_c, gqdval, gqdvec_a, num_blocks, num_threads);
	print_gqdvector_dev(gqdvec_c);

	// ----------
	// (a, b)
	// ----------
	// DD
	ip_qdvector(&qdval, qdvec_a, qdvec_b);
	printf(" DD : (a, b) = ");
	cout.precision(qd_real::_ndigits);
	cout << qdval << "\n";

	// GQD
	ip_gqdvector_dev(ptr_gqdval_dev, gqdvec_a, gqdvec_b, num_blocks, num_threads);
	//ip_gqdvector_dev(ptr_gqdval_dev, gqdvec_a, gqdvec_b, 1, num_threads);
	gqd2qd_dev(&qdval, ptr_gqdval_dev);
	printf("GQD : (a, b) = ");
	cout.precision(qd_real::_ndigits);
	cout << qdval << "\n";

	// ----------
	// ||a||_2
	// ----------
	// DD
	norm2_qdvector(&qdval, qdvec_a);
	printf(" DD : ||a||_2 = ");
	cout.precision(qd_real::_ndigits);
	cout << qdval << "\n";

	// GQD
	norm2_gqdvector_dev(ptr_gqdval_dev, gqdvec_a, num_blocks, num_threads);
	//norm2_gqdvector_dev(ptr_gqdval_dev, gqdvec_a, 1, num_threads);
	gqd2qd_dev(&qdval, ptr_gqdval_dev);
	printf("GQD : ||a||_2 = ");
	cout.precision(qd_real::_ndigits);
	cout << qdval << "\n";

/* Matrix */
	// DD & GQD
	qdmat_a = init_qdmatrix(ROW_DIM, COL_DIM);
	qdmat_b = init_qdmatrix(ROW_DIM, COL_DIM);
	qdmat_c = init_qdmatrix(ROW_DIM, COL_DIM);

	gqdmat_a = init_gqdmatrix_dev(ROW_DIM, COL_DIM);
	gqdmat_b = init_gqdmatrix_dev(ROW_DIM, COL_DIM);
	gqdmat_c = init_gqdmatrix_dev(ROW_DIM, COL_DIM);

	for(i = 0; i < ROW_DIM; i++)
	{
		for(j = 0; j < COL_DIM; j++)
		{
			// val := sqrt((i + 1));
			qdmat_a->element[i * COL_DIM + j] = (qd_real)sqrt((qd_real)(i + j + 1));
			qdmat_b->element[i * COL_DIM + j] = (qd_real)sqrt((qd_real)(i + j + 1));
		}
	}
	subst_gqdmatrix_dev_qdmat(gqdmat_a, qdmat_a);
	subst_gqdmatrix_dev_qdmat(gqdmat_b, qdmat_b);

	// Print gqdmatrix
	printf("qdmat_a:\n");
	normf_qdmatrix(&qdval, qdmat_a);
	cout.precision(qd_real::_ndigits);
	cout << qdval << "\n";

	printf("gqdmat_a:\n");
	//print_gqdmatrix_dev(gqdmat_a);
	normf_gqdmatrix_dev(ptr_gqdval_dev, gqdmat_a, 1, num_threads);
	gqd2qd_dev(&qdval, ptr_gqdval_dev);
	cout.precision(qd_real::_ndigits);
	cout << qdval << "\n";

	printf("qdmat_b:\n");
	normf_qdmatrix(&qdval, qdmat_b);
	cout.precision(qd_real::_ndigits);
	cout << qdval << "\n";

	printf("gqdmat_b:\n");
	//print_gqdmatrix_dev(gqdmat_b);
	normf_gqdmatrix_dev(ptr_gqdval_dev, gqdmat_b, num_blocks, num_threads);
	gqd2qd_dev(&qdval, ptr_gqdval_dev);
	cout.precision(qd_real::_ndigits);
	cout << qdval << "\n";

	// ----------
	// C := A * B
	// ----------
	// QD
	stime = get_real_secv();
	mul_qdmatrix(qdmat_c, qdmat_a, qdmat_b);
	etime[2] = get_real_secv() - stime;

	printf("QD : || A * B ||_F: %10.3g\n", etime[2]);
	normf_qdmatrix(&qdval, qdmat_c);
	cout.precision(qd_real::_ndigits);
	cout << qdval << "\n";

	// GQD
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	stime = get_real_secv();
	mul_gqdmatrix_dev(gqdmat_c, gqdmat_a, gqdmat_b, num_blocks, num_threads);
	etime[3] = get_real_secv() - stime;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cuda_etime[3], start, stop); // Unit: ms
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("GQD: || A * B ||_F: %10.3g, %10.3g\n", etime[3], cuda_etime[3] / 1000.0);
	normf_gqdmatrix_dev(ptr_gqdval_dev, gqdmat_c, num_blocks, num_threads);
	gqd2qd_dev(&qdval, ptr_gqdval_dev);
	cout.precision(qd_real::_ndigits);
	cout << qdval << "\n";

/* Free! */
	cudaFree(ptr_gqdval_dev);
	free_gqdvector_dev(gqdvec_a);
	free_gqdvector_dev(gqdvec_b);
	free_gqdvector_dev(gqdvec_c);

	free_gqdmatrix_dev(gqdmat_a);
	free_gqdmatrix_dev(gqdmat_b);
	free_gqdmatrix_dev(gqdmat_c);

	// GQD end!
	GQDEnd();

	// Free QDVectors
	free_qdvector(qdvec_a);
	free_qdvector(qdvec_b);
	free_qdvector(qdvec_c);

	free_qdmatrix(qdmat_a);
	free_qdmatrix(qdmat_b);
	free_qdmatrix(qdmat_c);

	return 0;
}